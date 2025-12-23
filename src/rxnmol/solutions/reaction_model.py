import os
import sys
import logging
import re
import math
import time
import copy
import gc
from typing import List, Tuple, Dict, Any, Union
import heapq

import torch
import torch.nn as nn

import torch.onnx
import torch.jit
import torch.utils.checkpoint
import sentencepiece as spm





logger = logging.getLogger(__name__)
# Note: Logging handlers are configured in run.py via logging.basicConfig()
# Do NOT add handlers here - it causes duplicate log messages!


pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3

seq_len = 100

num_heads = 8
num_layers = 8
d_model = 512
d_ff = 2048
d_k = d_model // num_heads
drop_out_rate = 0.2
device = torch.device('cuda') #if torch.cuda.is_available() else torch.device('cpu')

def pad_or_truncate(tokenized_text: List[int], max_len: int = 100) -> List[int]:
    if len(tokenized_text) > max_len:
        return tokenized_text[:max_len]
    else:
        return tokenized_text + [0] * (max_len - len(tokenized_text))


class RxnPredictor:
    def __init__(
        self,
        model_dir,
        checkpoint_path=None,
        model_path=None,
        device='cuda',
        tokenizer=None,
        batch_size=128, #128,  # Max batch size for predictions
        enforce_cuda=True,
        ):
             
        self.device = device
        self.batch_size = batch_size
        self.src_sp = spm.SentencePieceProcessor(model_file=f'{model_dir}/tokenizer_src.model')
        self.trg_sp = spm.SentencePieceProcessor(model_file=f'{model_dir}/tokenizer_trg.model')
        self.tokenizer = tokenizer or self.smiles_tokenizer
        
        # Optimized model loading priority:
        # 1. Compiled model (fastest) - loads in ~2s
        # 2. Full model.pt (fast) - loads in ~1s, but needs compilation (~60s)
        # 3. Checkpoint (slower) - needs reconstruction + compilation
        #
        # Note: Compiled models contain OptimizedModule and require weights_only=False
        # in PyTorch 2.6+. This is safe since we created the model ourselves.
        
        # Use a new name for the KV-cache enabled compiled model to avoid loading old incompatible ones
        compiled_path = f'{model_dir}/model_compiled_kv.pt'
        model_path = f'{model_dir}/model.pt'
        checkpoint_path = f'{model_dir}/atom_mit_checkpoint_last.pt'


        model_loaded = False
        # load_start = logger.info("Loading reaction prediction model...")

        t0 = time.time()
 
        logger.info(f"RxnPredictor: Loading model on device='{device}'")
        if enforce_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but CUDA is not available. "
                    "Please check your PyTorch installation and GPU drivers."
                )

        # Priority 1: Compiled model (fastest - already optimized)
        # IMPORTANT: Only try compiled model on CUDA - TensorRT models don't work on CPU
        if device == 'cuda' and os.path.exists(compiled_path):
            try:
                logger.info(f"Loading pre-compiled model from {compiled_path}")
                # PyTorch 2.6 requires weights_only=False for compiled models (contains OptimizedModule)
                # This is safe since we created this file ourselves
                self.model = torch.load(
                    compiled_path,
                    map_location=torch.device(device),
                    weights_only=False,
                    mmap=False,
                    )
                model_loaded = True
                logger.info(f"✓ Loaded compiled model in {time.time()-t0:.2f}s")
            except Exception as e:
                logger.warning(f"Failed to load compiled model (will try fallback): {e}")

        if device != 'cuda':
            logger.info(f"Running on CPU - skipping TensorRT-compiled model")

        # Priority 2: Full model.pt (fast) - fallback if compiled model failed or on CPU
        if not model_loaded and model_path and os.path.exists(model_path):
            logger.info(f"Loading full model from {model_path}")
            self.model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
            model_loaded = True
            logger.info(f"✓ Loaded model.pt in {time.time()-t0:.2f}s")

        # Priority 3: Checkpoint (needs model reconstruction)
        if not model_loaded and checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading from checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
            self.model = Transformer(
                src_vocab_size=self.src_sp.get_piece_size(),
                trg_vocab_size=self.trg_sp.get_piece_size(),
                d_model=d_model,
                n_heads=num_heads,
                num_layers=num_layers,
                d_ff=d_ff,
                max_seq_len=seq_len,
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            model_loaded = True
            logger.info(f"✓ Loaded from checkpoint in {time.time()-t0:.2f}s")

        if not model_loaded:
            raise ValueError(
                f"No valid model found. Searched for:\n"
                f"  1. {compiled_path} (compiled, fastest)\n"
                f"  2. {model_path} (full model)\n"
                f"  3. {checkpoint_path or 'N/A'} (checkpoint)\n"
                f"Please provide one of these files."
            )
        
        self.model.eval()
        
        # Compile if not already compiled
        if not os.path.exists(compiled_path) and torch.cuda.is_available():

            # if torch.cuda.is_available():
            import torch_tensorrt
            logger.info("Compiling model with TensorRT (first run only, ~30-60s)...")
            t_compile = time.time()
            # self.model = torch_tensorrt.compile(
            #     self.model, 
            #     inputs=[torch_tensorrt.Input((1, seq_len), dtype=torch.int32)],
            #     enabled_precisions={torch.float, torch.half},  # Run with FP16
            # )
            self.model = torch.compile(self.model, backend='tensorrt')
            logger.info(f"✓ Model compiled in {time.time()-t_compile:.2f}s")
                            
            # Save compiled model for future runs
            try:
                logger.info(f"Saving compiled model to {compiled_path}")
                torch.save(self.model, compiled_path)
                logger.info(f"✓ Compiled model saved. Future loads will be ~30x faster!")
            except Exception as e:
                logger.warning(f"Failed to save compiled model: {e}")
        
        logger.info(f"Model ready. Total load time: {time.time()-t0:.2f}s")

        if device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but CUDA is not available. "
                    "Please check your PyTorch installation and GPU drivers."
                )
            gpu_name = torch.cuda.get_device_name(0)
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using GPU: {gpu_name}")
            logger.info(f"GPU Memory: {total_mem_gb:.2f} GB")

            # Auto-configure batch size based on GPU memory

            if total_mem_gb < 12: # e.g. 11GB (2080Ti)
                self.batch_size = 1024
            elif total_mem_gb < 26: # e.g. 24GB (3090/4090/A5000)
                self.batch_size = 2048
            else: # e.g. 48GB (A6000) or 80GB (A100)
                self.batch_size = 4096
            logger.info(f"Auto-configured batch size: {self.batch_size} (based on {total_mem_gb:.1f}GB memory)")
        else:
            # CPU mode: use conservative batch size to avoid OOM
            # CPU inference is much slower and memory-intensive due to KV-cache
            # KV-cache grows with batch_size * seq_len * num_layers, and accumulates
            # Batch size 16 is safe for systems with 8GB+ RAM
            self.batch_size = 16
            logger.warning(
                f"Running on CPU with batch_size={self.batch_size}. "
                f"CPU inference is ~100x slower than GPU. Consider using CUDA if available."
            )

        
    def predict(self, input_text: str, is_tokenized=False) -> str:
        # logger.debug(f"input_text: {input_text}")
        if not is_tokenized:
            tokenized_input = self.tokenizer(input_text)
        else:
            tokenized_input = input_text
        logger.debug(f"tokenized_input: {tokenized_input}")

        src_tokenized = self.src_sp.EncodeAsIds(tokenized_input)
        src_tokenized = pad_or_truncate(src_tokenized + [eos_id])
        # with torch.no_grad():
        #     self.model.eval()
        with torch.inference_mode():
            src_j = torch.LongTensor([src_tokenized]).to(self.device)
            # print(f"src_j: {src_j.shape}")
            
            encoder_mask = (src_j != pad_id).unsqueeze(1).to(self.device)
            # print(f"encoder_mask: {encoder_mask.shape}")
            src_j = self.model.src_embedding(src_j)
            src_j = self.model.positional_encoder(src_j)
            encoder_output = self.model.encoder(src_j, encoder_mask)
            # print(f"encoder_output: {encoder_output.shape}")
                
            self.last_prediction = greedy_search(self.model, encoder_output, encoder_mask, self.trg_sp).replace(' ', '')
            logger.debug(f"Rxn Predicted: {input_text} >> {self.last_prediction}")
        return self.last_prediction.replace(' ', '')

    def predict_batch(self, input_texts: List[str]) -> List[str]:
        """
        Predict products for a batch of reaction inputs.

        Args:
            input_texts: List of reactant pairs ("reactant1.reactant2")

        Returns:
            List of predicted product SMILES
        """

        t_start = time.time()
        total_count = len(input_texts)
        logger.debug(f"🔬 GPU Batch Prediction: {total_count} reactions (max batch size: {self.batch_size})")

        predictions = self._predict_batch_with_oom_retry(input_texts)

        elapsed = time.time() - t_start
        throughput = total_count / elapsed if elapsed > 0 else 0
        logger.debug(f"✓ GPU prediction complete: {total_count} reactions in {elapsed:.3f}s ({throughput:.1f} rxn/s)")

        return predictions

    def _predict_batch_with_oom_retry(self, input_texts: List[str]) -> List[str]:
        """
        Predict batch with automatic OOM recovery.

        On CUDA OOM error:
        1. Clear CUDA cache
        2. Reduce batch size by half
        3. Retry with smaller chunks
        4. Update self.batch_size for future calls

        Args:
            input_texts: Reactions to predict
            batch_size: Current batch size to try

        Returns:
            List of predicted products
        """
        # Minimum batch size for OOM recovery (don't reduce below this)
        # On CPU, use a smaller minimum since memory is the constraint
        min_batch_size = 8 if self.device == 'cpu' else 128
        current_batch_size = min(self.batch_size, len(input_texts))

        # For small batches (fewer inputs than batch size), run directly
        if len(input_texts) <= self.batch_size:
            return self._predict_batch_single(input_texts)

        while current_batch_size >= min_batch_size:
            try:
                logger.debug(
                    f"GPU predict attempt: {len(input_texts)} inputs @ batch_size={current_batch_size} (default={self.batch_size})"
                )

                if len(input_texts) <= current_batch_size:
                    # Single batch
                    result = self._predict_batch_single(input_texts)
                    # Only update batch_size if we had previously reduced it due to OOM
                    # (current_batch_size < self.batch_size means we're in recovery mode)
                    # Don't update if current_batch_size == self.batch_size (normal operation)
                    return result
                else:
                    # Multiple smaller chunks
                    all_predictions = []
                    for i in range(0, len(input_texts), current_batch_size):
                        chunk = input_texts[i:i+current_batch_size]
                        predictions = self._predict_batch_single(chunk)
                        all_predictions.extend(predictions)
                        # Force garbage collection on CPU between chunks to free KV-cache memory
                        if self.device == 'cpu':
                            gc.collect()
                    # Don't update self.batch_size here - only reduce on OOM errors
                    return all_predictions

            except Exception as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    logger.warning(f"CUDA OOM encountered at batch_size={current_batch_size} for {len(input_texts)} inputs: {e}")
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    mem_msg = ""
                    mem_stats = self._get_cuda_mem_stats()
                    if mem_stats:
                        mem_msg = (
                            f" (alloc={mem_stats['allocated']:.2f}GB, "
                            f"reserved={mem_stats['reserved']:.2f}/{mem_stats['total']:.2f}GB)"
                        )

                    # Reduce batch size
                    # new_batch_size = current_batch_size // 2
                    new_batch_size = int(current_batch_size * 0.7)
                    logger.warning(
                        f"⚠️ CUDA OOM at batch_size={current_batch_size} for {len(input_texts)} inputs{mem_msg}; "
                        f"reducing {current_batch_size} to {new_batch_size} and retrying..."
                    )

                    # Update instance batch size for future calls
                    if new_batch_size >= min_batch_size:
                        self.batch_size = new_batch_size
                        logger.info(f"Updated default batch_size to {new_batch_size}")

                    current_batch_size = new_batch_size
                else:
                    # Not an OOM error, re-raise
                    logger.error(
                        f"Batch prediction failed (non-OOM) at batch_size={current_batch_size} for {len(input_texts)} inputs: {e}"
                    )
                    raise

        # If we get here, even min_batch_size failed
        # Try one-by-one as last resort
        logger.warning(f"⚠️ Batch prediction failed even at min_batch_size={min_batch_size}, falling back to sequential")
        results = []
        for text in input_texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Single prediction failed: {e}")
                results.append("FAILED")
        return results
    
    def _predict_batch_single(self, input_texts: List[str]) -> List[str]:
        """
        Internal method to predict a single batch (no chunking).
        """
        batch_size = len(input_texts)
        
        # Tokenize
        t_tokenize = time.time()
        tokenized_inputs = [self.tokenizer(text) for text in input_texts]
        
        # Encode as IDs
        src_tokenized_list = [self.src_sp.EncodeAsIds(t) + [eos_id] for t in tokenized_inputs]
        
        # Pad
        padded_src = []
        for t in src_tokenized_list:
            if len(t) > seq_len:
                t = t[:seq_len]
            else:
                t = t + [pad_id] * (seq_len - len(t))
            padded_src.append(t)
        
        tokenize_time = time.time() - t_tokenize
        
        t_gpu = time.time()    
        
        # TODO: Remove memory tracking in future (development only)
        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        src_tensor = torch.LongTensor(padded_src).to(self.device) # (B, L)
        
        with torch.inference_mode():
            encoder_mask = (src_tensor != pad_id).unsqueeze(1).to(self.device) # (B, 1, L)
            
            src_emb = self.model.src_embedding(src_tensor)
            src_emb = self.model.positional_encoder(src_emb)
            encoder_output = self.model.encoder(src_emb, encoder_mask)
            
            predictions = greedy_search_batch(self.model, encoder_output, encoder_mask, self.trg_sp)
        
        gpu_time = time.time() - t_gpu
        total_time = time.time() - t_tokenize
        
        # TODO: Remove memory tracking in future (development only)
        mem_used = 0
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
        
        logger.debug(f"    Batch({batch_size}): tokenize={tokenize_time*1000:.1f}ms, gpu={gpu_time*1000:.1f}ms, total={total_time*1000:.1f}ms, mem={mem_used:.1f}MB")
            
        return [p.replace(' ', '') for p in predictions]

    @staticmethod
    def _get_cuda_mem_stats():
        """Return simple CUDA memory stats for logging."""
        if not torch.cuda.is_available():
            return None
        try:
            idx = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(idx) / 1e9
            reserved = torch.cuda.memory_reserved(idx) / 1e9
            total = torch.cuda.get_device_properties(idx).total_memory / 1e9
            return {"allocated": allocated, "reserved": reserved, "total": total}
        except Exception:
            return None

    def smiles_tokenizer(self, smiles):
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smiles)]
        return ' '.join(tokens)    
    

def greedy_search(model, e_output, e_mask, trg_sp):
    device = e_output.device
    last_words = torch.LongTensor([pad_id] * seq_len).to(device) # (L)
    last_words[0] = sos_id # (L)
    cur_len = 1

    for i in range(seq_len):
        d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

        trg_embedded = model.trg_embedding(last_words.unsqueeze(0))
        trg_positional_encoded = model.positional_encoder(trg_embedded)
        decoder_output = model.decoder(
            trg_positional_encoded,
            e_output,
            e_mask,
            d_mask
        ) # (1, L, d_model)

        output = model.softmax(
            model.output_linear(decoder_output)
        ) # (1, L, trg_vocab_size)

        output = torch.argmax(output, dim=-1) # (1, L)
        last_word_id = output[0][i].item()

        if i < seq_len-1:
            last_words[i+1] = last_word_id
            cur_len += 1

        if last_word_id == eos_id:
            break

    if last_words[-1].item() == pad_id:
        decoded_output = last_words[1:cur_len].tolist()
    else:
        decoded_output = last_words[1:].tolist()
    decoded_output = trg_sp.decode_ids(decoded_output)

    return decoded_output

def greedy_search_batch(model, e_output, e_mask, trg_sp):
    batch_size = e_output.size(0)
    device = e_output.device
    
    # Initialize output tensor (B, L)
    last_words = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)
    last_words[:, 0] = sos_id
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Initialize KV cache
    # List of None for each layer initially
    past_key_values = [None] * model.num_layers
    
    # Current input token (B, 1)
    current_input = last_words[:, 0].unsqueeze(1) # (B, 1)
    
    for i in range(seq_len - 1):
        # Create decoder mask
        # For cached decoding, we only process the current token (length 1).
        # It attends to all past tokens (via cache) and itself.
        # The mask should allow attending to everything in the past.
        # Since we are doing greedy search, we attend to everything generated so far.
        # So mask is effectively all 1s (or None) because we want to see everything.
        # But wait, standard implementation expects mask shape to match attention scores.
        # Attn scores: (B, H, 1, L_past+1).
        # Mask: (B, 1, 1, L_past+1).
        # Since we want to attend to everything, we can just pass None or all True.
        # The padding mask is still relevant if we had padding in the prefix, but here
        # we are generating, so everything is valid (except maybe finished sequences).
        
        # Actually, we need to mask out pad tokens if we were doing batch padding,
        # but here we are generating.
        # Let's use None for d_mask in cached mode as we want full attention to history.
        d_mask = None
        
        # Embed current token
        trg_embedded = model.trg_embedding(current_input) # (B, 1, d_model)
        
        # Positional Encoding
        # We need to apply the correct positional encoding for position `i`.
        # model.positional_encoder adds PE[0:len]. We need PE[i:i+1].
        # We can manually fetch it from the buffer.
        pe = model.positional_encoder.positional_encoding[:, i:i+1, :].to(device)
        trg_positional_encoded = trg_embedded * math.sqrt(model.d_model) + pe
        
        # Run Decoder with Cache
        decoder_output, past_key_values = model.decoder(
            trg_positional_encoded,
            e_output,
            e_mask,
            d_mask,
            past_key_values=past_key_values,
            use_cache=True
        ) # (B, 1, d_model)
        
        output = model.softmax(
            model.output_linear(decoder_output)
        ) # (B, 1, trg_vocab_size)
        
        # Get prediction for current step
        predictions = torch.argmax(output, dim=-1) # (B, 1)
        next_words = predictions.squeeze(1) # (B,)
        
        last_words[:, i+1] = next_words
        
        # Update current input for next step
        current_input = next_words.unsqueeze(1)
        
        is_eos = (next_words == eos_id)
        finished = finished | is_eos
        
        if finished.all():
            break
            
    # Decode
    decoded_outputs = []
    for j in range(batch_size):
        words = last_words[j].tolist()
        try:
            eos_idx = words.index(eos_id)
            words = words[1:eos_idx]
        except ValueError:
            words = words[1:]
            
        decoded_outputs.append(trg_sp.decode_ids(words))
        
    return decoded_outputs


def beam_search(model, e_output, e_mask, trg_sp, beam_size=3):
    device = e_output.device
    cur_queue = PriorityQueue()
    #for k in range(beam_size):
    cur_queue.put(BeamNode(sos_id, -0.0, [sos_id]))

    finished_count = 0
    for pos in range(seq_len):
        new_queue = PriorityQueue()
        for k in range(beam_size):
            if pos == 0 and k > 0:
                continue
            else:
                node = cur_queue.get()

            if node.is_finished:
                new_queue.put(node)
            else:
                trg_input = torch.LongTensor(node.decoded + [pad_id] * (seq_len - len(node.decoded))).to(device) # (L)
                d_mask = (trg_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
                nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
                nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
                d_mask = d_mask & nopeak_mask # (1, L, L) padding false

                trg_embedded = model.trg_embedding(trg_input.unsqueeze(0))
                trg_positional_encoded = model.positional_encoder(trg_embedded)
                decoder_output = model.decoder(
                    trg_positional_encoded,
                    e_output,
                    e_mask,
                    d_mask
                ) # (1, L, d_model)

                output = model.softmax(
                    model.output_linear(decoder_output)
                ) # (1, L, trg_vocab_size)

                output = torch.topk(output[0][pos], dim=-1, k=beam_size)
                last_word_ids = output.indices.tolist() # (k)
                last_word_prob = output.values.tolist() # (k)

                for i, idx in enumerate(last_word_ids):
                    new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                    if idx == eos_id:
                        #new_node.prob = new_node.prob / float(len(new_node.decoded))
                        new_node.is_finished = True
                        finished_count += 1
                    new_queue.put(new_node)

        cur_queue = copy.deepcopy(new_queue)

        #if finished_count == beam_size:
        #    break

    #decoded_output = cur_queue.get().decoded
    #if decoded_output[-1] == eos_id:
    #    decoded_output = decoded_output[1:-1]
    #else:
    #    decoded_output = decoded_output[1:]
    #return trg_sp.decode_ids(decoded_output)
    all_candidates = list()
    scores  = [ ]
    for _ in range(beam_size):
        node = cur_queue.get()
        decoded_output = node.decoded
        scores.append(node.prob)
        all_candidates.append(trg_sp.decode_ids(decoded_output))

    return all_candidates, scores


class BeamNode():
    def __init__(self, cur_idx, prob, decoded):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False

    def __gt__(self, other):
        return self.prob > other.prob

    def __ge__(self, other):
        return self.prob >= other.prob

    def __lt__(self, other):
        return self.prob < other.prob

    def __le__(self, other):
        return self.prob <= other.prob

    def __eq__(self, other):
        return self.prob == other.prob

    def __ne__(self, other):
        return self.prob != other.prob

    def print_spec(self):
        print(f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}")


class PriorityQueue():

    def __init__(self):
        self.queue = []

    def put(self, obj):
        heapq.heappush(self.queue, (obj.prob, obj))

    def get(self):
        return heapq.heappop(self.queue)[1]

    def qsize(self):
        return len(self.queue)

    def print_scores(self):
        scores = [t[0] for t in self.queue]
        print(scores)

    def print_objs(self):
        objs = [t[1] for t in self.queue]
        print(objs)


class Transformer(nn.Module):
    """
    A Transformer model following the architecture described in "Attention Is All You Need".
    It consists of an Encoder and a Decoder stacked in layers, with embeddings and
    positional encoding at the input side and a linear + softmax at the output side.
    """
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
        max_seq_len: int = 512,
        *args, **kwargs
    ):
        """
        Args:
            src_vocab_size (int): Size of the source vocabulary.
            trg_vocab_size (int): Size of the target vocabulary.
            d_model (int): Dimensionality of embeddings and hidden states.
            n_heads (int): Number of parallel attention heads.
            num_layers (int): Number of Encoder/Decoder layers.
            d_ff (int): Hidden dimensionality of the feed-forward layers.
            dropout_rate (float): Dropout rate for all dropout layers.
            max_seq_len (int): Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len

        # Embeddings
        self.src_embedding = nn.Embedding(self.src_vocab_size, self.d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, self.d_model)

        # Positional Encoding
        self.positional_encoder = PositionalEncoder(
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
        )

        # Encoder and Decoder stacks
        self.encoder = Encoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate
        )

        self.decoder = Decoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate
        )

        # Final linear + softmax
        self.output_linear = nn.Linear(self.d_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None, past_key_values=None):
        """
        Forward pass of the Transformer model.

        Args:
            src_input (Tensor): Source input tokens of shape (batch_size, src_seq_len).
            trg_input (Tensor): Target input tokens of shape (batch_size, trg_seq_len).
            e_mask (Tensor): Optional encoder attention mask of shape (batch_size, src_seq_len) 
                             or (batch_size, src_seq_len, src_seq_len).
            d_mask (Tensor): Optional decoder attention mask of shape (batch_size, trg_seq_len) 
                             or (batch_size, trg_seq_len, trg_seq_len).
            past_key_values (List[Tuple]): Optional cache for fast decoding.

        Returns:
            Tensor of shape (batch_size, trg_seq_len, trg_vocab_size)
            Optional List[Tuple] if past_key_values is provided
        """
        # Embed inputs
        # Note: If using cache, trg_input is usually length 1 (current token)
        # src_input is still full length (needed for encoder if not cached, but here we assume encoder runs fully)
        # Actually, if we use cache, we usually pass pre-computed e_output to decoder, 
        # but this class structure runs encoder every time.
        # To support proper caching, we should allow passing e_output directly or 
        # assume src_input is full.
        
        # For greedy_search_batch optimization, we will call model.decoder directly there.
        # But if we want to use model(..., past_key_values=...), we need to handle it.
        
        src_embedded = self.src_embedding(src_input)  # (B, src_len, d_model)
        trg_embedded = self.trg_embedding(trg_input)  # (B, trg_len, d_model)

        # Positional encoding
        # If using cache, trg_encoded needs correct position offset!
        # But PositionalEncoder implementation adds PE based on 0..len(x).
        # If x is length 1 (step 50), it will get PE(0), which is WRONG.
        # We need to fix PositionalEncoder or handle it here.
        
        # Since we can't easily change PositionalEncoder without breaking weights (buffer registration),
        # we should handle the offset logic in greedy_search by slicing the PE buffer manually
        # or by passing full sequence to PE and slicing.
        
        # Let's assume for now this forward() is for training (no cache) 
        # and we use manual calls in greedy_search for inference.
        
        src_encoded = self.positional_encoder(src_embedded)
        trg_encoded = self.positional_encoder(trg_embedded)

        # Encoder and Decoder
        e_output = self.encoder(src_encoded, e_mask)          # (B, src_len, d_model)
        
        if past_key_values is not None:
            d_output, present_key_values = self.decoder(trg_encoded, e_output, e_mask, d_mask, past_key_values=past_key_values)
        else:
            d_output = self.decoder(trg_encoded, e_output, e_mask, d_mask)

        # Projection to vocabulary
        logits = self.output_linear(d_output)  # (B, trg_len, trg_vocab_size)
        output = self.softmax(logits)          # (B, trg_len, trg_vocab_size)
        
        if past_key_values is not None:
            return output, present_key_values
        return output


class Encoder(nn.Module):
    """
    The Transformer Encoder stack. Consists of N layers of self-attention 
    + feed-forward blocks, each preceded by LayerNorm and accompanied by dropouts.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        dropout_rate: float
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = LayerNormalization(d_model=d_model)

    def forward(self, x, e_mask=None):
        """
        Args:
            x (Tensor): Input embeddings to encode, shape (B, L, d_model).
            e_mask (Tensor): Optional encoder mask, shape can be (B, L) or (B, L, L).

        Returns:
            Tensor of shape (B, L, d_model)
        """
        for layer in self.layers:
            x = layer(x, e_mask)
        return self.layer_norm(x)


class Decoder(nn.Module):
    """
    The Transformer Decoder stack. Consists of N layers that include
    masked self-attention, encoder-decoder attention, and feed-forward blocks.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        dropout_rate: float
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = LayerNormalization(d_model=d_model)

    def forward(self, x, e_output, e_mask=None, d_mask=None, past_key_values=None, use_cache=False):
        """
        Args:
            x (Tensor): Target input embeddings, shape (B, L, d_model).
            e_output (Tensor): Encoder output, shape (B, src_len, d_model).
            e_mask (Tensor): Optional mask for encoder attention.
            d_mask (Tensor): Optional mask for decoder self-attention.
            past_key_values (List[Tuple]): List of past key values for each layer.
            use_cache (bool): Whether to return new key values.

        Returns:
            Tensor of shape (B, L, d_model)
            Optional List[Tuple] of new past key values
        """
        present_key_values = []
        
        for i, layer in enumerate(self.layers):
            if past_key_values is not None or use_cache:
                layer_past = past_key_values[i] if past_key_values is not None else None
                x, layer_past = layer(x, e_output, e_mask, d_mask, layer_past=layer_past, use_cache=use_cache)
                present_key_values.append(layer_past)
            else:
                x = layer(x, e_output, e_mask, d_mask)
                
        if past_key_values is not None or use_cache:
            return self.layer_norm(x), present_key_values
        return self.layer_norm(x)


class EncoderLayer(nn.Module):
    """
    One layer of the Transformer Encoder, containing:
    1) Self-attention sub-layer
    2) Feed-forward sub-layer
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model=d_model)
        self.multihead_attention = MultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate
        )
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.layer_norm_2 = LayerNormalization(d_model=d_model)
        self.feed_forward = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate
        )
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x, e_mask=None):
        # Self-attention sub-layer
        x_norm = self.layer_norm_1(x)
        attn_out = self.multihead_attention(x_norm, x_norm, x_norm, mask=e_mask)
        x = x + self.dropout_1(attn_out)

        # Feed-forward sub-layer
        x_norm = self.layer_norm_2(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout_2(ff_out)
        return x


class DecoderLayer(nn.Module):
    """
    One layer of the Transformer Decoder, containing:
    1) Masked self-attention sub-layer
    2) Encoder-decoder attention sub-layer
    3) Feed-forward sub-layer
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(d_model=d_model)
        self.masked_multihead_attention = MultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate
        )
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.layer_norm_2 = LayerNormalization(d_model=d_model)
        self.multihead_attention = MultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate
        )
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.layer_norm_3 = LayerNormalization(d_model=d_model)
        self.feed_forward = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate
        )
        self.dropout_3 = nn.Dropout(dropout_rate)

    def forward(self, x, e_output, e_mask=None, d_mask=None, layer_past=None, use_cache=False):
        # layer_past is (self_attn_past, cross_attn_past) or just self_attn_past?
        # Let's just cache self-attention for now as it's the main autoregressive cost.
        # Cross-attention inputs (e_output) are static, but re-projecting them is costly.
        # However, handling cross-attn cache requires more complex logic (reuse vs append).
        # Let's stick to Self-Attention Cache for simplicity and safety.
        
        self_attn_past = layer_past
        
        # Masked self-attention
        x_norm = self.layer_norm_1(x)
        
        if self_attn_past is not None or use_cache:
            masked_attn_out, self_attn_present = self.masked_multihead_attention(
                x_norm, x_norm, x_norm, mask=d_mask, past_key_value=self_attn_past, use_cache=True
            )
        else:
            masked_attn_out = self.masked_multihead_attention(x_norm, x_norm, x_norm, mask=d_mask)
            self_attn_present = None # Or we could return it if we wanted to start caching
            
        x = x + self.dropout_1(masked_attn_out)

        # Encoder-decoder attention
        x_norm = self.layer_norm_2(x)
        enc_dec_attn_out = self.multihead_attention(x_norm, e_output, e_output, mask=e_mask)
        x = x + self.dropout_2(enc_dec_attn_out)

        # Feed-forward
        x_norm = self.layer_norm_3(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout_3(ff_out)
        
        if layer_past is not None or use_cache:
            return x, self_attn_present
        return x


class MultiheadAttention(nn.Module):
    """
    Multi-head attention module with projection layers.
    """
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head
        self.dropout_rate = dropout_rate
        self.scale = math.sqrt(self.d_k)

        # Projection layers for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, past_key_value=None, use_cache=False):
        """
        Args:
            q, k, v (Tensor): Query, Key, and Value tensors of shape (B, L, d_model).
            mask (Tensor): Optional mask for attention of shape (B, L) or (B, L, L).
            past_key_value (Tuple[Tensor, Tensor]): Optional cached (k_proj, v_proj) from previous step.
            use_cache (bool): Whether to return the current key/value for caching.

        Returns:
            Tensor of shape (B, L, d_model)
            Optional Tuple[Tensor, Tensor] if past_key_value is not None or use_cache is True
        """
        bsz, seq_len, _ = q.size()

        # Linear projections
        q_proj = self.w_q(q).view(bsz, seq_len, self.n_heads, self.d_k)
        k_proj = self.w_k(k).view(bsz, -1, self.n_heads, self.d_k)
        v_proj = self.w_v(v).view(bsz, -1, self.n_heads, self.d_k)

        # Transpose to (B, n_heads, L, d_k)
        q_proj = q_proj.transpose(1, 2)
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        # KV Cache Logic
        if past_key_value is not None:
            past_k, past_v = past_key_value
            
            # Check if we are doing Cross-Attention (static K, V) or Self-Attention (growing K, V)
            # Heuristic: If input k length is same as past k length, it's likely static (Cross-Attn reuse)
            # But usually in decoding, input k for self-attn is length 1 (current token).
            
            # For this implementation, we will assume:
            # 1. If it's Self-Attention, we append.
            # 2. If it's Cross-Attention, we should ideally reuse, but standard Transformer 
            #    implementations often just re-project or expect the caller to handle static inputs.
            #    To be safe and simple: We will ALWAYS append if provided, assuming the caller
            #    manages the inputs correctly (i.e. for self-attn, pass only new token).
            
            k_proj = torch.cat([past_k, k_proj], dim=2)
            v_proj = torch.cat([past_v, v_proj], dim=2)
            
        current_key_value = (k_proj, v_proj)

        # Scaled dot-product attention
        attn_values = self.scaled_dot_product_attention(q_proj, k_proj, v_proj, mask=mask)

        # Reshape back to (B, L, d_model)
        attn_values = attn_values.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        # Final linear projection
        output = self.w_o(attn_values)
        
        if past_key_value is not None or use_cache:
            return output, current_key_value
        return output

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # q, k, v are of shape (B, n_heads, L, d_k)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale  # (B, n_heads, L, L)

        if mask is not None:
            # Broadcase mask to match attn_scores shape
            # mask can be (B, L) => unsqueeze to (B, 1, 1, L)
            # or mask can be (B, L, L) => unsqueeze to (B, 1, L, L)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, L, L)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.attn_softmax(attn_scores)  # (B, n_heads, L, L)
        attn_weights = self.attention_dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)       # (B, n_heads, L, d_k)
        return attn_out


class FeedForwardLayer(nn.Module):
    """
    Position-wise feed-forward layer:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class LayerNormalization(nn.Module):
    """
    Layer normalization with learnable parameters.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)


class PositionalEncoder(nn.Module):
    """
    Positional encoding module that encodes positional information into embeddings 
    using sine and cosine functions of different frequencies, as described in the paper.
    """
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Create a long enough PEs
        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer so it doesn't get updated by backprop
        self.register_buffer("positional_encoding", pe.unsqueeze(0))  # shape (1, max_seq_len, d_model)

    def forward(self, x):
        """
        Adds positional encoding to the input x.

        Args:
            x (Tensor): Input of shape (B, L, d_model)

        Returns:
            Tensor of shape (B, L, d_model) with added positional information.
        """
        # Scale embedding by sqrt(d_model) for improved stability
        x = x * math.sqrt(self.d_model)

        # Add positional encodings (pe might be bigger than x's sequence length)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        return x
