#pragma once

#include <string>
#include <memory>
#include <algorithm>

#include "common/vocab.h"
#include "common/encoder.h"
#include "common/decoder.h"
#include "common/model.h"
#include "common/utils.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>


class NMTDecoder {
 public:
   NMTDecoder(
       std::shared_ptr<Weights> model,
       std::shared_ptr<Vocab> srcVocab,
       std::shared_ptr<Vocab> trgVocab)
       : model_(model),
         srcVocab_(srcVocab),
         trgVocab_(trgVocab),
         encoder_(new Encoder(*model_)),
         decoder_(new Decoder(*model_)) {
   }

   std::pair<std::vector<std::string>, float> translate(std::string& sentence) {
     prepareSourceSentence(sentence);
     prepareDecoder();

     float score = 0.00f;
     std::vector<std::string> translation;
     while (true) {
       decoder_->GetProbs(Probs_, AlignedSourceContext_,
                          PrevState_, PrevEmbedding_, SourceContext_);
       auto maxIter = thrust::max_element(Probs_.begin(), Probs_.end());
       float prob = *maxIter;
       size_t word_index = std::distance(Probs_.begin(), maxIter);

       score += prob;
       if (word_index == (*trgVocab_)["</s>"]) {
         break;
       }
       translation.push_back((*trgVocab_)[word_index]);
       std::vector<size_t> batch(1, word_index);
       decoder_->Lookup(Embedding_, batch);
       decoder_->GetNextState(State_, Embedding_,
                              PrevState_, AlignedSourceContext_);

       mblas::Swap(State_, PrevState_);
       mblas::Swap(Embedding_, PrevEmbedding_);
     }

     std::pair<std::vector<std::string>, float> result(translation, score);
     return result;
   }
 private:
   void prepareSourceSentence(std::string& sentence) {
     Trim(sentence);
     std::vector<std::string> tokens;
     Split(sentence, tokens, " ");
     auto encoded_tokens = srcVocab_->Encode(tokens, true);
     encoder_->GetContext(encoded_tokens, SourceContext_);
   }

   void prepareDecoder() {
     decoder_->EmptyState(PrevState_, SourceContext_, 1);
     decoder_->EmptyEmbedding(PrevEmbedding_, 1);
   }

 protected:
   std::shared_ptr<Weights> model_;
   std::shared_ptr<Vocab> srcVocab_;
   std::shared_ptr<Vocab> trgVocab_;
   std::shared_ptr<Encoder> encoder_;
   std::shared_ptr<Decoder> decoder_;
   mblas::Matrix SourceContext_;
   mblas::Matrix PrevState_;
   mblas::Matrix PrevEmbedding_;

   mblas::Matrix AlignedSourceContext_;
   mblas::Matrix Probs_;

   mblas::Matrix State_;
   mblas::Matrix Embedding_;

};
