// $Id$

#ifndef moses_DynamicCacheBasedLanguageModel_h
#define moses_DynamicCacheBasedLanguageModel_h

#include "FeatureFunction.h"

#ifdef WITH_THREADS
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/locks.hpp>
#endif

typedef std::pair<int, float> decaying_cache_value_t; 
typedef std::map<std::string, decaying_cache_value_t > decaying_cache_t; 

#define CBLM_QUERY_TYPE_ALLSUBSTRINGS 0
#define CBLM_QUERY_TYPE_WHOLESTRING 1

#define CBLM_SCORE_TYPE_HYPERBOLA 0
#define CBLM_SCORE_TYPE_POWER 1
#define CBLM_SCORE_TYPE_EXPONENTIAL 2
#define CBLM_SCORE_TYPE_COSINE 3
#define CBLM_SCORE_TYPE_HYPERBOLA_REWARD 10
#define CBLM_SCORE_TYPE_POWER_REWARD 11
#define CBLM_SCORE_TYPE_EXPONENTIAL_REWARD 12
#define PI 3.14159265

namespace Moses
{

class WordsRange;

/** Calculates score for the Dynamic Cache-Based pesudo LM
 */
class DynamicCacheBasedLanguageModel : public StatelessFeatureFunction
{
  // data structure for the cache;
  // the key is the word and the value is the decaying score
  decaying_cache_t m_cache;
  size_t query_type; //way of querying the cache
  size_t score_type; //way of scoring entries of the cache
  std::string m_initfiles; // vector of files loaded in the initialization phase

  std::vector<float> precomputedScores;
  unsigned int maxAge;

#ifdef WITH_THREADS
  //multiple readers - single writer lock
  mutable boost::shared_mutex m_cacheLock;
#endif

  float decaying_score(int age);
  void SetPreComputedScores();

  float Evaluate_Whole_String( const TargetPhrase&) const;
  float Evaluate_All_Substrings( const TargetPhrase&) const;

  void Decay();
  void Update(std::vector<std::string> words, int age);
        
  void Execute(std::vector<std::string> commands);
  void Execute_Single_Command(std::string command);
  
  void Load(std::vector<std::string> files);
  void Load_Single_File(const std::string file);

  void Insert(std::vector<std::string> ngrams);

  void Evaluate( const TargetPhrase&, ScoreComponentCollection* ) const;

  void Print() const;

  void Clear();
  void SetQueryType(size_t type);
  void SetScoreType(size_t type);
  void SetMaxAge(unsigned int age);

public:
  DynamicCacheBasedLanguageModel(const std::string &line);
  ~DynamicCacheBasedLanguageModel();

  bool IsUseable(const FactorMask &mask) const {
    return true;
  }

  void Load();
  void Load(const std::string file);
  void Execute(std::string command);
  void SetParameter(const std::string& key, const std::string& value);

  void Insert(std::string &entries);

  void Evaluate(const Phrase &source
                        , const TargetPhrase &targetPhrase
                        , ScoreComponentCollection &scoreBreakdown
                        , ScoreComponentCollection &estimatedFutureScore) const;

};


/*
class DynamicCacheBasedLanguageModel : public StatelessFeatureFunction
{
  // data structure for the cache;
  // the key is the word and the value is the decaying score
  decaying_cache_t m_cache;
  size_t query_type; //way of querying the cache
  size_t score_type; //way of scoring entries of the cache
  std::string m_initfiles; // vector of files loaded in the initialization phase

  float decaying_score(int age);
  float Evaluate_Whole_String( const TargetPhrase&) const;
  float Evaluate_All_Substrings( const TargetPhrase&) const;
//  void Evaluate_Whole_String( const TargetPhrase&, ScoreComponentCollection* ) const;
//  void Evaluate_All_Substrings( const TargetPhrase&, ScoreComponentCollection* ) const;

  void Decay();
  void Update(std::vector<std::string> words, int age);
	
  void Execute(std::vector<std::string> commands);
  void Execute_Single_Command(std::string command);

  void Load(std::vector<std::string> files);
  void Load_Single_File(const std::string file);

  void Insert(std::vector<std::string> ngrams);

  void Print() const;
  void Clear();
  void Evaluate( const TargetPhrase&, ScoreComponentCollection* ) const;

public:
	
  DynamicCacheBasedLanguageModel(const std::string &line);
  ~DynamicCacheBasedLanguageModel();

  bool IsUseable(const FactorMask &mask) const {
    return true;
  }

  void Load();
  void Load(const std::string file);
  void Execute(std::string command);
  void SetParameter(const std::string& key, const std::string& value);

  inline void SetQueryType(size_t type) { query_type = type; };
  void Insert(std::string &entries);
//  void Evaluate(const PhraseBasedFeatureContext& context, ScoreComponentCollection* accumulator) const;
//  void EvaluateChart(const ChartBasedFeatureContext& context, ScoreComponentCollection* accumulator) const;

  void Evaluate(const Phrase &source
                        , const TargetPhrase &targetPhrase
                        , ScoreComponentCollection &scoreBreakdown
                        , ScoreComponentCollection &estimatedFutureScore) const;

};
*/


}

#endif