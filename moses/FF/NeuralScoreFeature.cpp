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
    NeuralScoreState(
        const amunmt::States& state,
        const std::deque<std::string>& context,
        const std::vector<std::string>& lastWords)
    : m_state(state),
      m_lastWord(lastWords.back()),
      m_lastContext(context)
    {
      for (size_t i = 0; i  < lastWords.size(); ++i)
        m_lastContext.push_back(lastWords[i]);
    }

    NeuralScoreState(amunmt::States state)
    : m_state(state),
      m_lastWord("")
    {}

    NeuralScoreState()
    : m_lastWord("")
    {}

    std::string ToString() const {
      std::stringstream ss;
      for(size_t i = 0; i < m_lastContext.size(); ++i) {
        if(i != 0) {
          ss << " ";
        }
        ss << m_lastContext[i];
      }
      return ss.str();
    }

  int Compare(const FFState& other) const
  {
    const NeuralScoreState &otherState = static_cast<const NeuralScoreState&>(other);
    if (m_lastContext.size() == otherState.m_lastContext.size() &&
          std::equal(m_lastContext.begin(), m_lastContext.end(),
                     otherState.m_lastContext.begin())) {
        return 0;
    }
    return (std::lexicographical_compare(m_lastContext.begin(), m_lastContext.end(),
                              otherState.m_lastContext.begin(),
                                                otherState.m_lastContext.end())) ? -1 : +1;
  }

  bool operator==(const FFState& other) const {
    if (Compare(other)) return true;
    else return false;
  }

  size_t hash() const {
    return 0;
  }

  void LimitLength(size_t length) {
    while(m_lastContext.size() > length)
      m_lastContext.pop_front();
  }

  const std::string& GetLastWord() const {
    return m_lastWord;
  }

  const std::deque<std::string>& GetContext() const {
    return m_lastContext;
  }

  const amunmt::States& GetState() const {
    return m_state;
  }

  private:
    amunmt::States m_state;
    std::string m_lastWord;
    std::deque<std::string> m_lastContext;
};



NeuralScoreFeature::NeuralScoreFeature(const std::string &line)
  : StatefulFeatureFunction(1, line),
    m_stateLength(5),
    m_factor(0),
    m_mode("rescore"),
    m_threadId(0)
{
  ReadParameters();
}

void NeuralScoreFeature::Load(AllOptions::ptr const& opts) {
  m_options = opts;
  amunmt::NMT::InitGod(m_configFilePath);

  size_t totalThreads = amunmt::NMT::GetTotalThreads();
  m_batchSize = amunmt::NMT::GetBatchSize();

  std::cerr << "Total amuNMT threads/scorers: " << totalThreads << std::endl;
}



void NeuralScoreFeature::InitializeForInput(ttasksptr const& ttask)
{
  if (!m_nmt.get()) {
    boost::mutex::scoped_lock lock(m_mutex);
    std::cerr << "NMT pointer is empty: creating nmt object" << std::endl;
    m_nmt.reset(new amunmt::NMT());
  } else {
    // m_nmt->SetDevice();
  }
}


void NeuralScoreFeature::CleanUpAfterSentenceProcessing(ttasksptr const& ttask)
{
}


const FFState* NeuralScoreFeature::EmptyHypothesisState(const InputType &input) const
{
  UTIL_THROW_IF2(input.GetType() != SentenceInput,
                 "This feature function requires the Sentence input type");
  const Sentence& sentence = static_cast<const Sentence&>(input);

  std::vector<std::string> sourceSentence;
  for (size_t i = 0; i < sentence.GetSize(); ++i) {
    sourceSentence.push_back(sentence.GetWord(i).GetString(0).as_string());
  }

  amunmt::States firstStates = m_nmt->CalcSourceContext(sourceSentence);


  return new NeuralScoreState(firstStates);
}


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


void NeuralScoreFeature::EvaluateTranslationOptionListWithSourceContext(
    const InputType &input,
    const TranslationOptionList &translationOptionList) const
{
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
  const amunmt::States& prevStates = static_cast<const NeuralScoreState*>(prev_state)->GetState();

  NeuralScoreState* newState =
    new NeuralScoreState(prevStates,
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

std::vector<float> NeuralScoreFeature::RescoreNBestList(
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


std::vector<amunmt::NeuralExtention> NeuralScoreFeature::RescoreStack(std::vector<Hypothesis*>& hyps, size_t index)
{
  std::vector<std::vector<std::string>> phrases;
  std::vector<amunmt::States> states;
  std::vector<float> probs(hyps.size(), 0.0f);

  for (auto& hyp : hyps) {
    if (hyp->GetId() == 0) {
        return std::vector<amunmt::NeuralExtention>();
    }

    std::vector<std::string> phrase;
    const TargetPhrase& tp = hyp->GetCurrTargetPhrase();

    for (size_t j = 0; j < tp.GetSize(); ++j) {
      phrase.push_back(tp.GetWord(j).GetString(m_factor).as_string());
    }

    if(hyp->IsSourceCompleted()) {
        phrase.push_back("</s>");
    }

    phrases.emplace_back(std::move(phrase));

    const NeuralScoreState* nState =
        static_cast<const NeuralScoreState*>(hyp->GetPrevHypo()->GetFFState(index));

    states.push_back(nState->GetState());
  }

  m_nmt->RescorePhrases(phrases, states, probs);
  // auto newHyps = m_nmt->ExtendHyps(states);

  // for (size_t i = 0; i < hyps.size(); ++i) {
    // const Hypothesis* prevHyp = hyps[i]->GetPrevHypo();
    // const NeuralScoreState* prevState =
         // static_cast<const NeuralScoreState*>(prevHyp->GetFFState(index));

    // const TargetPhrase& tp = hyps[i]->GetCurrTargetPhrase();
    // std::vector<std::string> phrase;
    // for(size_t j = 0; j < tp.GetSize(); ++j) {
      // phrase.push_back(tp.GetWord(j).GetString(m_factor).as_string());
    // }

    // NeuralScoreState* nState = new NeuralScoreState(states[i],
                                                    // prevState->GetContext(),
                                                    // phrase);

    // Scores scores(1);
    // scores[0] = probs[i];

    // ScoreComponentCollection& accumulator = hyps[i]->GetCurrScoreBreakdown();
    // accumulator.PlusEquals(this, scores);
    // hyps[i]->Recalc();

    // const NeuralScoreState* temp =
        // static_cast<const NeuralScoreState*>(hyps[i]->GetFFState(index));

    // hyps[i]->SetFFState(index, nState);
    // delete temp;
  // }
  std::vector<amunmt::NeuralExtention> newHyps;
  return newHyps;
}


void NeuralScoreFeature::ProcessStack(Collector& collector, size_t index)
{
  if (m_mode != "precalculate")
    return;
  UTIL_THROW_IF2(true,
                 "Only stack rescoring is supported.");

}

NeuralScoreFeature::~NeuralScoreFeature() {
  amunmt::NMT::Clean();
}

}

