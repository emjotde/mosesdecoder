#pragma once

#include <string>
#include <set>
#include <map>

#include "moses/SearchNormal.h"
#include "moses/HypothesisStackNormal.h"
#include "moses/TranslationOptionCollection.h"
#include "StatefulFeatureFunction.h"
#include "FFState.h"
#include <boost/shared_ptr.hpp>
#include <boost/thread/tss.hpp>

#include "common/scorer.h"

typedef std::vector<std::string> Prefix;

namespace amunmt {

class NMT;

}

namespace Moses {

class NeuralScoreFeature : public StatefulFeatureFunction {
 public:
  NeuralScoreFeature(const std::string &line);

  bool IsUseable(const FactorMask &mask) const {
    return true;
  }

  void InitializeForInput(ttasksptr const& ttask);
  void CleanUpAfterSentenceProcessing(ttasksptr const& ttask);

  void RescoreStack(std::vector<Hypothesis*>& hyps, size_t index);
  void RescoreStackBatch(std::vector<Hypothesis*>& hyps, size_t index);
  void ProcessStack(Collector& collector, size_t index);

  virtual const FFState* EmptyHypothesisState(const InputType &input) const;

  void EvaluateInIsolation(const Phrase &source
                           , const TargetPhrase &targetPhrase
                           , ScoreComponentCollection &scoreBreakdown
                           , ScoreComponentCollection &estimatedFutureScore) const;

  void EvaluateWithSourceContext(const InputType &input
                                 , const InputPath &inputPath
                                 , const TargetPhrase &targetPhrase
                                 , const StackVec *stackVec
                                 , ScoreComponentCollection &scoreBreakdown
                                 , ScoreComponentCollection *estimatedFutureScore = NULL) const;

  void EvaluateTranslationOptionListWithSourceContext(const InputType &input
      , const TranslationOptionList &translationOptionList) const;

  FFState* EvaluateWhenApplied(
    const Hypothesis& cur_hypo,
    const FFState* prev_state,
    ScoreComponentCollection* accumulator) const;

  FFState* EvaluateWhenApplied(
    const ChartHypothesis& /* cur_hypo */,
    int /* featureID - used to index the state in the previous hypotheses */,
    ScoreComponentCollection* accumulator) const;

  void SetParameter(const std::string& key, const std::string& value);

  std::vector<double> RescoreNBestList(std::vector<std::string> nbestList) const;


 private:
  std::string m_configFilePath;
  size_t m_batchSize;
  size_t m_stateLength;
  size_t m_factor;
  std::string m_mode;

  std::vector< std::vector<amunmt::ScorerPtr> > m_scorers;

  boost::thread_specific_ptr<amunmt::NMT> m_nmt;
  boost::thread_specific_ptr<std::set<std::string> > m_targetWords;

  size_t m_threadId;
  boost::mutex m_mutex;
};

}

