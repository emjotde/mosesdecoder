#ifndef moses_SearchCubePruning_h
#define moses_SearchCubePruning_h

#include <vector>
#include "Search.h"
#include "HypothesisStackCubePruning.h"
#include "SentenceStats.h"

namespace Moses
{
using NeuroPhraseColl = std::map<const Bitmap*, std::vector<TranslationOption*>>;

class InputType;
class TranslationOptionCollection;

class SkeletonPT;

class SearchCubePruning;

class FunctorCube {
  public:
    FunctorCube(SearchCubePruning* search) : m_search(search) {}

    virtual void operator()(const Bitmap &bitmap,
                            const Range &range,
                            BitmapContainer &bitmapContainer) = 0;

    virtual bool IsCollector() = 0;

  protected:
    SearchCubePruning* m_search;
};

class ExpanderCube : public FunctorCube {
  public:
    ExpanderCube(SearchCubePruning* search) : FunctorCube(search) {}
    virtual void operator()(const Bitmap &bitmap,
                            const Range &range,
                            BitmapContainer &bitmapContainer);

    virtual bool IsCollector() { return false; }
};

class CollectorCube : public FunctorCube, public Collector {
  public:
    CollectorCube(SearchCubePruning* search) : FunctorCube(search) {}
    virtual void operator()(const Bitmap &bitmap,
                            const Range &range,
                            BitmapContainer &bitmapContainer);

    virtual bool IsCollector() { return true; }

    std::vector<const Hypothesis*> GetHypotheses() {
      return m_hypotheses;
    }

    std::vector<const TranslationOptionList*>& GetOptions(int hypId) {
      return m_options[hypId];
    }

  private:
    std::vector<const Hypothesis*> m_hypotheses;
    std::map<size_t, std::vector<const TranslationOptionList*> > m_options;
};

/** Functions and variables you need to decoder an input using the phrase-based decoder with cube-pruning
 *  Instantiated by the Manager class
 */
class SearchCubePruning: public Search
{
protected:
  friend ExpanderCube;
  friend CollectorCube;
  std::map<Range, InputPath*> m_inputPathsMap;
  const PhraseDictionary* m_pt;
  std::vector<TranslationOption*> m_dynamicOptions;
  std::vector<TranslationOptionList*> m_transOptionLists;

  std::vector < HypothesisStack* > m_hypoStackColl; /**< stacks to store hypotheses (partial translations) */
  // no of elements = no of words in source + 1
  const TranslationOptionCollection &m_transOptColl; /**< pre-computed list of translation options for the phrases in this sentence */

  //! go thru all bitmaps in 1 stack & create backpointers to bitmaps in the stack
  void CreateForwardTodos(HypothesisStackCubePruning &stack, NeuroPhraseColl* neuro=nullptr);

  //! create a back pointer to this bitmap, with edge that has this words range translation
  void CreateForwardTodos(const Bitmap &bitmap, const Range &range,
                          BitmapContainer &bitmapContainer);

  void CreateForwardTodos(Bitmap const& newBitmap, Range const& range,
                          BitmapContainer& bitmapContainer,
                          const std::vector<TranslationOption*>& neuroPhrases);

  std::vector<Hypothesis*> CacheForNeural(Collector& collector);

  NeuroPhraseColl ProcessStackForNeuro(HypothesisStackCubePruning*& stack);


  //void CreateForwardTodos2(HypothesisStackCubePruning &stack);
  //! create a back pointer to this bitmap, with edge that has this words range translation
  //void CreateForwardTodos2(const WordsBitmap &bitmap, const Range &range, BitmapContainer &bitmapContainer);

  bool CheckDistortion(const Bitmap &bitmap, const Range &range) const;

  void PrintBitmapContainerGraph();

  void CleanAfterDecode();

public:
  SearchCubePruning(Manager& manager, const TranslationOptionCollection &transOptColl);
  ~SearchCubePruning();

  void Decode();

  void OutputHypoStackSize();
  void OutputHypoStack(int stack);

  virtual const std::vector < HypothesisStack* >& GetHypothesisStacks() const;
  virtual const Hypothesis *GetBestHypothesis() const;
};


}
#endif
