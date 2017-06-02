#include "Manager.h"
#include "Util.h"
#include "SearchCubePruning.h"
#include "StaticData.h"
#include "InputType.h"
#include "TranslationOptionCollection.h"
#include <boost/foreach.hpp>
#include "FF/NeuralScoreFeature.h"
#include "moses/TranslationModel/SkeletonPT.h"

using namespace std;

namespace Moses
{
class BitmapContainerOrderer
{
  public:

  bool operator()(const BitmapContainer* A, const BitmapContainer* B) const {
    if (B->Empty()) {
      if (A->Empty()) {
        return A < B;
      }
      return false;
    }
    if (A->Empty()) {
      return true;
    }

    // Compare the top hypothesis of each bitmap container using the TotalScore, which includes future cost
    const float scoreA = A->Top()->GetHypothesis()->GetFutureScore();
    const float scoreB = B->Top()->GetHypothesis()->GetFutureScore();

    if (scoreA < scoreB) {
      return true;
    } else if (scoreA > scoreB) {
      return false;
    } else {
      // Equal scores: break ties by comparing target phrases (if they exist)
      // *Important*: these are pointers to copies of the target phrases from the
      // hypotheses.  This class is used to keep priority queues ordered in the
      // background, so comparisons made as those data structures are cleaned up
      // may occur *after* the target phrases in hypotheses have been cleaned up,
      // leading to segfaults if relying on hypotheses to provide target phrases.
      boost::shared_ptr<TargetPhrase> phrA = A->Top()->GetTargetPhrase();
      boost::shared_ptr<TargetPhrase> phrB = B->Top()->GetTargetPhrase();
      if (!phrA || !phrB) {
        // Fallback: compare pointers, non-deterministic sort
        return A < B;
      }
      return (phrA->Compare(*phrB) > 0);
    }
  }
};

void ExpanderCube::operator()(const Bitmap &bitmap,
                              const Range &range,
                              BitmapContainer &bitmapContainer) {
  m_search->CreateForwardTodos(bitmap, range, bitmapContainer);
}

void CollectorCube::operator()(const Bitmap &bitmap,
                               const Range &range,
                               BitmapContainer &bitmapContainer) {
  const TranslationOptionList* transOptList;
  transOptList = m_search->m_transOptColl.GetTranslationOptionList(range);

  if (!transOptList) return;

  BOOST_FOREACH(const Hypothesis* hypothesis, bitmapContainer.GetHypotheses()) {
    if(m_options.count(hypothesis->GetId()) == 0)
      m_hypotheses.push_back(hypothesis);
    m_options[hypothesis->GetId()].push_back(transOptList);
  }
}

SearchCubePruning::
SearchCubePruning(Manager& manager, TranslationOptionCollection const& transOptColl)
  : Search(manager)
  , m_hypoStackColl(manager.GetSource().GetSize() + 1)
  , m_transOptColl(transOptColl)
{
  std::vector < HypothesisStackCubePruning >::iterator iterStack;
  for (size_t ind = 0 ; ind < m_hypoStackColl.size() ; ++ind) {
    HypothesisStackCubePruning *sourceHypoColl = new HypothesisStackCubePruning(m_manager);
    sourceHypoColl->SetMaxHypoStackSize(m_options.search.stack_size);
    sourceHypoColl->SetBeamWidth(m_options.search.beam_width);

    m_hypoStackColl[ind] = sourceHypoColl;
  }

  for (auto path : m_transOptColl.GetInputPaths()) {
    m_inputPathsMap[path->GetWordsRange()] = path;
  }


}

SearchCubePruning::~SearchCubePruning()
{
  RemoveAllInColl(m_hypoStackColl);
  CleanAfterDecode();
}

// void SearchCubePruning::CacheForNeural(Collector& collector) {
  // const std::vector<const StatefulFeatureFunction*> &ffs = StatefulFeatureFunction::GetStatefulFeatureFunctions();
  // const StaticData &staticData = StaticData::Instance();
  // for (size_t i = 0; i < ffs.size(); ++i) {
    // const NeuralScoreFeature* nsf = dynamic_cast<const NeuralScoreFeature*>(ffs[i]);
    // if (nsf && !staticData.IsFeatureFunctionIgnored(*ffs[i]))
      // const_cast<NeuralScoreFeature*>(nsf)->ProcessStack(collector, i);
  // }
// }

/**
 * Main decoder loop that translates a sentence by expanding
 * hypotheses stack by stack, until the end of the sentence.
 */
void SearchCubePruning::Decode()
{
  const std::vector<FeatureFunction*> &ffs = FeatureFunction::GetFeatureFunctions();
  for (auto& ff : ffs) {
    PhraseDictionary* pt = dynamic_cast<PhraseDictionary*>(ff);
    if (pt != nullptr) {
      m_pt = pt;
      break;
    }
  }
  // initial seed hypothesis: nothing translated, no words produced
  const Bitmap &initBitmap = m_bitmaps.GetInitialBitmap();
  Hypothesis *hypo = new Hypothesis(m_manager, m_source, m_initialTransOpt, initBitmap, m_manager.GetNextHypoId());

  HypothesisStackCubePruning &firstStack
  = *static_cast<HypothesisStackCubePruning*>(m_hypoStackColl.front());
  firstStack.AddInitial(hypo);
  // Call this here because the loop below starts at the second stack.
  firstStack.CleanupArcList();

  CreateForwardTodos(firstStack);

  const size_t PopLimit = m_manager.options()->cube.pop_limit;
  VERBOSE(2,"Cube Pruning pop limit is " << PopLimit << std::endl);

  const size_t Diversity = m_manager.options()->cube.diversity;
  VERBOSE(2,"Cube Pruning diversity is " << Diversity << std::endl);
  VERBOSE(2,"Max Phrase length is "
          << m_manager.options()->search.max_phrase_length << std::endl);

  // go through each stack
  size_t stackNo = 1;
  std::vector < HypothesisStack* >::iterator iterStack;
  for (iterStack = m_hypoStackColl.begin() + 1 ; iterStack != m_hypoStackColl.end() ; ++iterStack) {
    // BOOST_FOREACH(HypothesisStack* hstack, m_hypoStackColl) {
    if (this->out_of_time()) return;

    HypothesisStackCubePruning* sourceHypoColl =
      static_cast<HypothesisStackCubePruning*>(*iterStack);

    // priority queue which has a single entry for each bitmap
    // container, sorted by score of top hyp
    std::priority_queue < BitmapContainer*, std::vector< BitmapContainer* >,
        BitmapContainerOrderer > BCQueue;

    _BMType::const_iterator bmIter;
    const _BMType &accessor = sourceHypoColl->GetBitmapAccessor();

    for(bmIter = accessor.begin(); bmIter != accessor.end(); ++bmIter) {
      // build the first hypotheses
      IFVERBOSE(2) {
        m_manager.GetSentenceStats().StartTimeOtherScore();
      }
      bmIter->second->InitializeEdges();
      IFVERBOSE(2) {
        m_manager.GetSentenceStats().StopTimeOtherScore();
      }
      m_manager.GetSentenceStats().StartTimeManageCubes();
      BCQueue.push(bmIter->second);
      m_manager.GetSentenceStats().StopTimeManageCubes();

    }

    // main search loop, pop k best hyps
    for (size_t numpops = 1; numpops <= PopLimit && !BCQueue.empty(); numpops++) {
      // get currently best hypothesis in queue
      m_manager.GetSentenceStats().StartTimeManageCubes();
      BitmapContainer *bc = BCQueue.top();
      BCQueue.pop();
      m_manager.GetSentenceStats().StopTimeManageCubes();
      IFVERBOSE(2) {
        m_manager.GetSentenceStats().AddPopped();
      }
      // push on stack and create successors
      IFVERBOSE(2) {
        m_manager.GetSentenceStats().StartTimeOtherScore();
      }
      bc->ProcessBestHypothesis();
      IFVERBOSE(2) {
        m_manager.GetSentenceStats().StopTimeOtherScore();
      }
      // if there are any hypothesis left in this specific container, add back to queue
      m_manager.GetSentenceStats().StartTimeManageCubes();
      if (!bc->Empty())
        BCQueue.push(bc);
      m_manager.GetSentenceStats().StopTimeManageCubes();
    }

    // ensure diversity, a minimum number of inserted hyps for each bitmap container;
    //    NOTE: diversity doesn't ensure they aren't pruned at some later point
    if (Diversity > 0) {
      IFVERBOSE(2) {
        m_manager.GetSentenceStats().StartTimeOtherScore();
      }
      for(bmIter = accessor.begin(); bmIter != accessor.end(); ++bmIter) {
        bmIter->second->EnsureMinStackHyps(Diversity);
      }
      IFVERBOSE(2) {
        m_manager.GetSentenceStats().StopTimeOtherScore();
      }
    }

    // the stack is pruned before processing (lazy pruning):
    VERBOSE(3,"processing hypothesis from next stack");
    IFVERBOSE(2) {
      m_manager.GetSentenceStats().StartTimeStack();
    }
    sourceHypoColl->PruneToSize(m_options.search.stack_size);
    auto neuroTranslationOptions = ProcessStackForNeuro(sourceHypoColl);

    VERBOSE(3,std::endl);
    sourceHypoColl->CleanupArcList();


    IFVERBOSE(2) {
      m_manager.GetSentenceStats().StopTimeStack();
    }

    IFVERBOSE(2) {
      m_manager.GetSentenceStats().StartTimeSetupCubes();
    }

    CreateForwardTodos(*sourceHypoColl, &neuroTranslationOptions);

    IFVERBOSE(2) {
      m_manager.GetSentenceStats().StopTimeSetupCubes();
    }

    stackNo++;
  }
}

NeuroPhraseColl SearchCubePruning::ProcessStackForNeuro(HypothesisStackCubePruning*& stack) {
  HypothesisStackCubePruning::iterator h;
  std::vector<Hypothesis*> hypsToRescore;
  for (h = stack->begin(); h != stack->end(); ++h) {
    hypsToRescore.push_back(*h);
  }

  const auto& ffs = StatefulFeatureFunction::GetStatefulFeatureFunctions();
  const StaticData &staticData = StaticData::Instance();

  NeuroPhraseColl neuroTranslationOptions;
  for (size_t i = 0; i < ffs.size(); ++i) {
    const NeuralScoreFeature* nsf = dynamic_cast<const NeuralScoreFeature*>(ffs[i]);
    if (nsf && !staticData.IsFeatureFunctionIgnored(*ffs[i])) {
      auto neuroHyps = const_cast<NeuralScoreFeature*>(nsf)->RescoreStack(hypsToRescore, i);

      for (auto& prop : neuroHyps) {
        std::cerr << prop << std::endl;

        Hypothesis* prevHyp = hypsToRescore[prop.prevIndex_];

        if(prevHyp->IsSourceCompleted()) break;

        if (prop.coverage_.empty()) continue;

        std::stringstream ss;
        ss << prop.phrase_[0];
        for (size_t wi = 1; wi < prop.phrase_.size(); ++wi) {
          ss << " " << prop.phrase_[wi];
        }

        Range range (prop.coverage_.front(), prop.coverage_.back());
        if (prevHyp->GetWordsBitmap().Overlap(range)) {
          continue;
        }

        const Phrase& sourcePhrase = m_inputPathsMap[range]->GetPhrase();
        TargetPhrase* targetPhrase = m_pt->CreateNeuralTargetPhrase(sourcePhrase, ss.str(),
                                                                    prop.score_);
        auto& bitMap = m_bitmaps.GetBitmap(prevHyp->GetWordsBitmap(), range);

        TranslationOption* tOptions = new TranslationOption(range, *targetPhrase);
        m_dynamicOptions.push_back(tOptions);
        tOptions->SetInputPath(*m_inputPathsMap[range]);
        neuroTranslationOptions[&bitMap].push_back(tOptions);
      }
    }
  }
  return neuroTranslationOptions;
}


void SearchCubePruning::CreateForwardTodos(HypothesisStackCubePruning &stack, NeuroPhraseColl* neuro)
{
  const _BMType &bitmapAccessor = stack.GetBitmapAccessor();
  size_t size = m_source.GetSize();

  stack.AddHypothesesToBitmapContainers();

  for (auto& iterAccessor : bitmapAccessor) {
    const Bitmap& bitmap = *iterAccessor.first;
    BitmapContainer &bitmapContainer = *iterAccessor.second;

    if (bitmapContainer.GetHypothesesSize() == 0) {
      // no hypothese to expand. don't bother doing it
      continue;
    }

    // Sort the hypotheses inside the Bitmap Container as they are being used by now.
    bitmapContainer.SortHypotheses();

    // check bitamp and range doesn't overlap
    size_t startPos, endPos;
    for (startPos = 0 ; startPos < size ; startPos++) {
      if (bitmap.GetValue(startPos)) {
        continue;
      }

      // not yet covered
      Range applyRange(startPos, startPos);
      if (CheckDistortion(bitmap, applyRange)) {
        const Bitmap &newBitmap = m_bitmaps.GetBitmap(bitmap, applyRange);
        if (neuro) {
          auto& neuroPhrases = (*neuro)[&newBitmap];
          CreateForwardTodos(newBitmap, applyRange, bitmapContainer, neuroPhrases);
        } else {
          CreateForwardTodos(newBitmap, applyRange, bitmapContainer);
        }
      }

      size_t maxSize = size - startPos;
      size_t maxSizePhrase = m_manager.options()->search.max_phrase_length;
      maxSize = std::min(maxSize, maxSizePhrase);
      for (endPos = startPos+1; endPos < startPos + maxSize; endPos++) {
        if (bitmap.GetValue(endPos)) {
          break;
        }

        Range applyRange(startPos, endPos);
        if (CheckDistortion(bitmap, applyRange)) {
          const Bitmap &newBitmap = m_bitmaps.GetBitmap(bitmap, applyRange);
          if (neuro) {
            auto& neuroPhrases = (*neuro)[&newBitmap];
            CreateForwardTodos(newBitmap, applyRange, bitmapContainer, neuroPhrases);
          } else {
            CreateForwardTodos(newBitmap, applyRange, bitmapContainer);
          }
        }
      }
    }
  }
}

void
SearchCubePruning::
CreateForwardTodos(Bitmap const& newBitmap, Range const& range,
                   BitmapContainer& bitmapContainer,
                   const std::vector<TranslationOption*>& neuroPhrases)
{

  size_t numCovered = newBitmap.GetNumWordsCovered();
  const auto* transOptList = m_transOptColl.GetTranslationOptionList(range);
  TranslationOptionList* extendedTransOptList = new TranslationOptionList();
  m_transOptionLists.push_back(extendedTransOptList);

  for (auto& option : *transOptList) {
    extendedTransOptList->Add(option);
  }

  for(auto& option : neuroPhrases) {
    extendedTransOptList->Add(option);
  }
  static TranslationOption::Better cmp;
  std::sort(extendedTransOptList->begin(), extendedTransOptList->end(), cmp);
  const SquareMatrix& estimatedScores = m_transOptColl.GetEstimatedScores();

  if (transOptList && transOptList->size() > 0) {
    HypothesisStackCubePruning& newStack =
      *static_cast<HypothesisStackCubePruning*>(m_hypoStackColl[numCovered]);

    newStack.SetBitmapAccessor(newBitmap, newStack, range, bitmapContainer,
                               estimatedScores, *extendedTransOptList);
  }
}

void
SearchCubePruning::
CreateForwardTodos(Bitmap const& newBitmap, Range const& range,
                   BitmapContainer& bitmapContainer)
{
  // const Bitmap &newBitmap = m_bitmaps.GetBitmap(bitmap, range);

  size_t numCovered = newBitmap.GetNumWordsCovered();
  // const TranslationOptionList* transOptList;
  const auto* transOptList = m_transOptColl.GetTranslationOptionList(range);
  const SquareMatrix& estimatedScores = m_transOptColl.GetEstimatedScores();

  if (transOptList && transOptList->size() > 0) {
    HypothesisStackCubePruning& newStack =
      *static_cast<HypothesisStackCubePruning*>(m_hypoStackColl[numCovered]);

    newStack.SetBitmapAccessor(newBitmap, newStack, range, bitmapContainer,
                               estimatedScores, *transOptList);
  }
}


bool
SearchCubePruning::
CheckDistortion(const Bitmap &hypoBitmap, const Range &range) const
{
  // since we check for reordering limits, its good to have that limit handy
  int maxDistortion = m_manager.options()->reordering.max_distortion;
  if (maxDistortion < 0) return true;

  // if there are reordering limits, make sure it is not violated
  // the coverage bitmap is handy here (and the position of the first gap)
  size_t const startPos = range.GetStartPos();
  size_t const endPos = range.GetEndPos();

  // if reordering constraints are used (--monotone-at-punctuation or xml),
  // check if passes all
  if (!m_source.GetReorderingConstraint().Check(hypoBitmap, startPos, endPos))
    return false;

  size_t const hypoFirstGapPos = hypoBitmap.GetFirstGapPos();
  // any length extension is okay if starting at left-most edge
  if (hypoFirstGapPos == startPos) return true;

  // starting somewhere other than left-most edge, use caution
  // the basic idea is this: we would like to translate a phrase starting
  // from a position further right than the left-most open gap. The
  // distortion penalty for the following phrase will be computed relative
  // to the ending position of the current extension, so we ask now what
  // its maximum value will be (which will always be the value of the
  // hypothesis starting at the left-most edge).  If this vlaue is than
  // the distortion limit, we don't allow this extension to be made.
  Range bestNextExtension(hypoFirstGapPos, hypoFirstGapPos);
  return (m_source.ComputeDistortionDistance(range, bestNextExtension)
          <= maxDistortion);
}

/**
 * Find best hypothesis on the last stack.
 * This is the end point of the best translation, which can be traced back from here
 */
Hypothesis const*
SearchCubePruning::
GetBestHypothesis() const
{
  //	const HypothesisStackCubePruning &hypoColl = m_hypoStackColl.back();
  const HypothesisStack &hypoColl = *m_hypoStackColl.back();
  return hypoColl.GetBestHypothesis();
}

/**
 * Logging of hypothesis stack sizes
 */
void
SearchCubePruning::
OutputHypoStackSize()
{
  std::vector < HypothesisStack* >::const_iterator iterStack = m_hypoStackColl.begin();
  TRACE_ERR( "Stack sizes: " << (int)(*iterStack)->size());
  for (++iterStack; iterStack != m_hypoStackColl.end() ; ++iterStack) {
    TRACE_ERR( ", " << (int)(*iterStack)->size());
  }
  TRACE_ERR( endl);
}

void SearchCubePruning::PrintBitmapContainerGraph()
{
  HypothesisStackCubePruning &lastStack = *static_cast<HypothesisStackCubePruning*>(m_hypoStackColl.back());
  const _BMType &bitmapAccessor = lastStack.GetBitmapAccessor();

  _BMType::const_iterator iterAccessor;
  for (iterAccessor = bitmapAccessor.begin(); iterAccessor != bitmapAccessor.end(); ++iterAccessor) {
    cerr << iterAccessor->first << endl;
    //BitmapContainer &container = *iterAccessor->second;
  }

}

/**
 * Logging of hypothesis stack contents
 * \param stack number of stack to be reported, report all stacks if 0
 */
void SearchCubePruning::OutputHypoStack(int stack)
{
  if (stack >= 0) {
    TRACE_ERR( "Stack " << stack << ": " << endl << m_hypoStackColl[stack] << endl);
  } else {
    // all stacks
    int i = 0;
    vector < HypothesisStack* >::iterator iterStack;
    for (iterStack = m_hypoStackColl.begin() ; iterStack != m_hypoStackColl.end() ; ++iterStack) {
      HypothesisStackCubePruning &hypoColl = *static_cast<HypothesisStackCubePruning*>(*iterStack);
      TRACE_ERR( "Stack " << i++ << ": " << endl << hypoColl << endl);
    }
  }
}

const std::vector < HypothesisStack* >& SearchCubePruning::GetHypothesisStacks() const
{
  return m_hypoStackColl;
}

void SearchCubePruning::CleanAfterDecode() {
  for (auto option : m_dynamicOptions) {
    if (option) {
      delete option;
    }
  }
  m_dynamicOptions.clear();

  for (auto list : m_transOptionLists) {
    if (list) {
      list->resize(0);
      delete list;
    }
  }
  // for (auto option : m_transOptionLists) {
    // if (option) {
      // delete option;
    // }
  // }
  // m_transOptionLists.clear();
}

}

