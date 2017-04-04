#include <vector>
#include <string>
#include <sstream>
#include <deque>
#include <list>
#include <algorithm>

#include <boost/foreach.hpp>

#include "moses/ScoreComponentCollection.h"
#include "moses/TargetPhrase.h"
#include "moses/Hypothesis.h"
#include "moses/FF/NeuralScoreFeature.h"
#include "util/string_piece.hh"

#include "plugin/nmt.h"
#include "common/scorer.h"

namespace Moses
{

class NeuralScoreState : public FFState
{
 public:
  NeuralScoreState(amunmt::States& states, const std::deque<std::string>& context)
    : m_states(states),
      m_lastContext(context)
  {}

  NeuralScoreState(amunmt::States states)
    : m_states(states)
  {}

  NeuralScoreState()
  {}

  std::string ToString() const {
    std::stringstream ss;
    for(size_t i = 0; i < m_lastContext.size(); ++i) {
      if(i != 0)
        ss << " ";
      ss << m_lastContext[i];
    }
    return ss.str();
  }

  int Compare(const FFState& other) const
  {
    const NeuralScoreState &otherState = static_cast<const NeuralScoreState&>(other);
    if (m_lastContext.size() == otherState.m_lastContext.size() &&
        std::equal(m_lastContext.begin(),
                   m_lastContext.end(),
                   otherState.m_lastContext.begin()))
      return 0;
    return (std::lexicographical_compare(m_lastContext.begin(), m_lastContext.end(),
                   otherState.m_lastContext.begin(),
                   otherState.m_lastContext.end())) ? -1 : +1;
  }

  bool operator==(const FFState& other) const {
    if (Compare(other)) {
      return true;
    } else {
      return false;
    }
  }

  size_t hash() const {
    return 0;
  }

  void LimitLength(size_t length) {
    while(m_lastContext.size() > length)
      m_lastContext.pop_front();
  }

  const std::deque<std::string>& GetContext() const {
    return m_lastContext;
  }

  amunmt::States GetState() const {
    return m_states;
  }

 private:
  amunmt::States m_states;
  std::deque<std::string> m_lastContext;
};


NeuralScoreFeature::NeuralScoreFeature(const std::string &line)
  : StatefulFeatureFunction(1, line),
    m_batchSize(1000),
    m_stateLength(5),
    m_factor(0),
    m_mode("rescore"),
    m_threadId(0)
{
  ReadParameters();
  amunmt::NMT::InitGod(m_configFilePath);

  size_t totalThreads = amunmt::NMT::GetTotalThreads();

  std::cerr << "TotalThreads:" << totalThreads << std::endl;

  for (size_t i = 0; i < totalThreads; ++i) {
    m_scorers.push_back(amunmt::NMT::NewScorers());
  }
}


void NeuralScoreFeature::InitializeForInput(ttasksptr const& ttask)
{
  if (!m_nmt.get()) {
    boost::mutex::scoped_lock lock(m_mutex);
    size_t device = m_threadId++ % m_scorers.size();
    m_nmt.reset(new amunmt::NMT(m_scorers[device]));
    m_targetWords.reset(new std::set<std::string>());
    m_nmt->SetDevice();
  } else {
    m_nmt.reset(new amunmt::NMT(m_nmt->GetScorers()));
    m_nmt->SetDevice();
  }
  m_nmt->ClearStates();
  m_targetWords->clear();
}


void NeuralScoreFeature::CleanUpAfterSentenceProcessing(ttasksptr const& ttask)
{
}


const FFState* NeuralScoreFeature::EmptyHypothesisState(const InputType &input) const
{
  UTIL_THROW_IF2(input.GetType() != SentenceInput,
                 "This feature function requires the Sentence input type");
  const Sentence& sentence = static_cast<const Sentence&>(input);

  // if (m_filteredSoftmax > 0) {
    // m_targetWords->insert("eos");
    // m_targetWords->insert("UNK");
    // m_nmt->FilterTargetVocab(*m_targetWords, m_filteredSoftmax);
  // }

  std::vector<std::string> sourceSentence;
  for (size_t i = 0; i < sentence.GetSize(); ++i) {
    sourceSentence.push_back(sentence.GetWord(i).GetString(0).as_string());
  }

  amunmt::States firstStates = m_nmt->CalcSourceContext(sourceSentence);

  return new NeuralScoreState(firstStates);
}


void NeuralScoreFeature::RescoreStack(std::vector<Hypothesis*>& hyps, size_t index)
{
  if(m_mode != "rescore")
    return;

  std::vector<Hypothesis*> batch;
  for (size_t i = 0; i < hyps.size(); ++i) {
    if (batch.size() < m_batchSize) {
      batch.push_back(hyps[i]);
    } else {
      RescoreStackBatch(batch, index);
      batch.clear();
      batch.push_back(hyps[i]);
    }
  }

  if(batch.size() > 0) {
    RescoreStackBatch(batch, index);
  }
}


void NeuralScoreFeature::RescoreStackBatch(std::vector<Hypothesis*>& hyps, size_t index)
{
  size_t maxLength = 1;
  bool firstWord = false;
  bool complete = false;
  BOOST_FOREACH(Hypothesis* hyp, hyps) {
    size_t tpLength = hyp->GetCurrTargetPhrase().GetSize();

    if(hyp->GetWordsBitmap().GetNumWordsCovered() == 0)
      firstWord = true;

    if(hyp->IsSourceCompleted())
      complete = true;

    if(tpLength > maxLength)
      maxLength = tpLength;
  }

  if (complete) {
    maxLength++;
  }

  Batches batches(maxLength, Batch(hyps.size(), -1));
  std::vector<amunmt::States> states(hyps.size());
  Scores probs(hyps.size(), 0.0);
  Scores unks(hyps.size(), 0.0);

  for (size_t i = 0; i < hyps.size(); ++i) {
    const TargetPhrase& tp = hyps[i]->GetCurrTargetPhrase();
    for (size_t j = 0; j < tp.GetSize(); ++j) {
      batches[j][i] = m_nmt->TargetVocab(tp.GetWord(j).GetString(m_factor).as_string());
    }

    if(complete) {
      batches[tp.GetSize()][i] = m_nmt->TargetVocab("</s>");
    }

    if(hyps[i]->GetId() == 0) {
      return;
    }
  }

  m_nmt->BatchSteps(batches, probs, unks, states);

  for (size_t i = 0; i < hyps.size(); ++i) {
    const Hypothesis* prevHyp = hyps[i]->GetPrevHypo();
    const NeuralScoreState* prevState = static_cast<const NeuralScoreState*>(prevHyp->GetFFState(index));
    const TargetPhrase& tp = hyps[i]->GetCurrTargetPhrase();
    std::vector<std::string> phrase;
    for(size_t j = 0; j < tp.GetSize(); ++j) {
      phrase.push_back(tp.GetWord(j).GetString(m_factor).as_string());
    }

    NeuralScoreState* nState = new NeuralScoreState(states[i], prevState->GetContext(), phrase);

    Scores scores(1);
    scores[0] = probs[i];
    //scores[1] = unks[i];

    ScoreComponentCollection& accumulator = hyps[i]->GetCurrScoreBreakdown();
    accumulator.PlusEquals(this, scores);
    hyps[i]->Recalc();

    const NeuralScoreState* temp = static_cast<const NeuralScoreState*>(hyps[i]->GetFFState(index));
    hyps[i]->SetFFState(index, nState);
    delete temp;
  }
}


void NeuralScoreFeature::ProcessStack(Collector& collector, size_t index)
{
  if (m_mode != "precalculate")
    return;
  UTIL_THROW_IF2(true,
                 "Only stack rescoring is supported.");

  // std::map<int, const NeuralScoreState*> states;

  // bool first = true;
  // size_t covered;
  // size_t total;

  // if(!m_pbl.get()) {
    // m_pbl.reset(new PrefsByLength());
  // }
  // m_pbl->clear();

  // BOOST_FOREACH(const Hypothesis* h, collector.GetHypotheses()) {
    // const Hypothesis& hypothesis = *h;

    // const FFState* ffstate = hypothesis.GetFFState(index);
    // const NeuralScoreState* state
      // = static_cast<const NeuralScoreState*>(ffstate);

    // if(first) {
      // const Bitmap& hypoBitmap = hypothesis.GetWordsBitmap();
      // covered = hypoBitmap.GetNumWordsCovered();
      // total = hypoBitmap.GetSize();
      // first = false;
    // }

    // size_t hypId = hypothesis.GetId();
    // states[hypId] = state;

    // BOOST_FOREACH(const TranslationOptionList* tol, collector.GetOptions(hypId)) {
      // TranslationOptionList::const_iterator iter;
      // for (iter = tol->begin() ; iter != tol->end() ; ++iter) {
        // const TranslationOption& to = **iter;
        // const TargetPhrase& tp = to.GetTargetPhrase();

        // Prefix prefix;
        // for(size_t i = 0; i < tp.GetSize(); ++i) {
          // prefix.push_back(tp.GetWord(i).GetString(m_factor).as_string());
          // if(m_pbl->size() < prefix.size())
            // m_pbl->resize(prefix.size());

          // (*m_pbl)[prefix.size() - 1][prefix][hypId] = Payload();
        // }
        // if(total - covered == to.GetSize()) {
          // prefix.push_back("eos");
          // if(m_pbl->size() < prefix.size())
            // m_pbl->resize(prefix.size());

          // (*m_pbl)[prefix.size() - 1][prefix][hypId] = Payload();
        // }
      // }
    // }
  // }

  // for(size_t l = 0; l < m_pbl->size(); l++) {
    // Prefixes& prefixes = (*m_pbl)[l];

    // std::vector<std::string> allWords;
    // std::vector<std::string> allLastWords;
    // std::vector<StateInfoPtr> allStates;

    // for(Prefixes::iterator it = prefixes.begin(); it != prefixes.end(); it++) {
      // const Prefix& prefix = it->first;
      // BOOST_FOREACH(SP& hyp, it->second) {
        // size_t hypId = hyp.first;
        // allWords.push_back(prefix[l]);
        // StateInfoPtr state;
        // if(prefix.size() == 1) {
          // state = states[hypId]->GetState();
          // allLastWords.push_back(states[hypId]->GetLastWord());
        // }
        // else {
          // Prefix prevPrefix = prefix;
          // prevPrefix.pop_back();
          // state = (*m_pbl)[prevPrefix.size() - 1][prevPrefix][hypId].state_;
          // allLastWords.push_back(prevPrefix.back());
        // }
        // allStates.push_back(state);
      // }
    // }

    // std::vector<double> allProbs;
    // std::vector<StateInfoPtr> allOutStates;
    // std::vector<bool> unks;

    // BatchProcess(allWords,
                 // allLastWords,
                 // allStates,
                 // [>* out *<]
                 // allProbs,
                 // allOutStates,
                 // unks);

    // size_t k = 0;
    // for(Prefixes::iterator it = prefixes.begin(); it != prefixes.end(); it++) {
      // BOOST_FOREACH(SP& hyp, it->second) {
        // Payload& payload = hyp.second;
        // payload.logProb_ = allProbs[k];
        // payload.state_ = allOutStates[k];
        // payload.known_ = unks[k];
        // k++;
      // }
    // }
  // }
}


// void NeuralScoreFeature::BatchProcess(
  // const std::vector<std::string>& nextWords,
  // const std::vector<StateInfoPtr>& inputStates,
  // std::vector<double>& logProbs,
  // std::vector<StateInfoPtr>& outputStates,
  // std::vector<bool>& unks)
// {
    // size_t items = nextWords.size();
    // size_t batches = ceil(items/(float)m_batchSize);
    // for(size_t i = 0; i < batches; ++i) {
      // size_t thisBatchStart = i * m_batchSize;
      // size_t thisBatchEnd = std::min(thisBatchStart + m_batchSize, items);


      // std::vector<std::string> nextWordsBatch(nextWords.begin() + thisBatchStart,
                                              // nextWords.begin() + thisBatchEnd);
      // std::vector<StateInfoPtr> inputStatesBatch(inputStates.begin() + thisBatchStart,
                                              // inputStates.begin() + thisBatchEnd);

      // std::vector<double> logProbsBatch;
      // std::vector<StateInfoPtr> nextStatesBatch;
      // std::vector<bool> unksBatch;

      // m_nmt->MakeStep(nextWordsBatch,
                // inputStatesBatch,
                // [>* out *<]
                // logProbsBatch,
                // nextStatesBatch,
                // unksBatch);

      // logProbs.insert(logProbs.end(), logProbsBatch.begin(), logProbsBatch.end());
      // outputStates.insert(outputStates.end(), nextStatesBatch.begin(), nextStatesBatch.end());
      // unks.insert(unks.end(), unksBatch.begin(), unksBatch.end());
    // }
// }


void NeuralScoreFeature::EvaluateInIsolation(const Phrase &source
    , const TargetPhrase &targetPhrase
    , ScoreComponentCollection &scoreBreakdown
    , ScoreComponentCollection &estimatedFutureScore) const
{
}


void NeuralScoreFeature::EvaluateWithSourceContext(const InputType &input
    , const InputPath &inputPath
    , const TargetPhrase &targetPhrase
    , const StackVec *stackVec
    , ScoreComponentCollection &scoreBreakdown
    , ScoreComponentCollection *estimatedFutureScore) const
{
}


void NeuralScoreFeature::EvaluateTranslationOptionListWithSourceContext(const InputType &input
    , const TranslationOptionList &translationOptionList) const
{
  // if(m_filteredSoftmax > 0) {
    // TranslationOptionList::const_iterator iter;
    // for (iter = translationOptionList.begin();
         // iter != translationOptionList.end(); ++iter) {
      // const TranslationOption& to = **iter;
      // const TargetPhrase& tp = to.GetTargetPhrase();

      // for(size_t i = 0; i < tp.GetSize(); ++i) {
        // std::string temp = tp.GetWord(i).GetString(m_factor).as_string();
        // const_cast<std::set<std::string>&>(*m_targetWords).insert(temp);
      // }
    // }
  // }
}


FFState* NeuralScoreFeature::EvaluateWhenApplied(
  const Hypothesis& cur_hypo,
  const FFState* prev_state,
  ScoreComponentCollection* accumulator) const
{
  std::vector<float> newScores(m_numScoreComponents, 0);

  const TargetPhrase& tp = cur_hypo.GetCurrTargetPhrase();
  Prefix phrase;

  for(size_t i = 0; i < tp.GetSize(); ++i) {
    std::string word = tp.GetWord(i).GetString(m_factor).as_string();
    phrase.push_back(word);
  }

  if(cur_hypo.IsSourceCompleted()) {
    phrase.push_back("</s>");
  }

  float prob = 0.0f;
  size_t unks = 0;
  const amunmt::States& prevStates = static_cast<const NeuralScoreState*>(prev_state)->GetStates();
  amunmt::States newStates;

  NeuralScoreState* newState =
    new NeuralScoreState(newStates
                         static_cast<const NeuralScoreState*>(prev_state)->GetContext(),
                         phrase);

  newScores[0] = prob;
  newScores[1] = unks;

  accumulator->PlusEquals(this, newScores);

  newState->LimitLength(m_stateLength);
  return newState;
}


FFState* NeuralScoreFeature::EvaluateWhenApplied(
  const ChartHypothesis& /* cur_hypo */,
  int /* featureID - used to index the state in the previous hypotheses */,
  ScoreComponentCollection* accumulator) const
{
  return new NeuralScoreState(amunmt::States());
}

std::vector<double> NeuralScoreFeature::RescoreNBestList(
    std::vector<std::string> nbestList) const
{
  return m_nmt->RescoreNBestList(nbestList);
}

void NeuralScoreFeature::SetParameter(const std::string& key, const std::string& value)
{
  if (key == "state-length") {
    m_stateLength = Scan<size_t>(value);
  } else if (key == "mode") {
    m_mode = value;
  } else if (key == "config-path") {
    m_configFilePath = value;
  } else {
    StatefulFeatureFunction::SetParameter(key, value);
  }
}

}

