import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch import nn
import argparse
import re  
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5EncoderModel
from tokenizers import Tokenizer
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Usage:
# python scripts/extract_T5.py -i newdata/s8754_wt.fasta -o newdata/ProstT5/wt_embeddings.pt
# python scripts/extract_T5.py -i newdata/s669/s669_mut.fasta -o newdata/s669/ProstT5/mut_embeddings.pt
# python scripts/extract_T5.py -i newdata/s669/s669_wt.fasta -o newdata/s669/ProstT5/wt_embeddings.pt

#python newdata/esmc_300m/pt_del.py  newdata/ProstT5/mut_embeddings.pt newdata/ProstT5/wt_embeddings.pt newdata/ProstT5/result.pt
#python newdata/esmc_300m/pt_del.py  newdata/s669/ProstT5/mut_embeddings.pt newdata/s669/ProstT5/wt_embeddings.pt newdata/s669/ProstT5/result.pt

# python scripts/reg-train-test.py  --train_input newdata/ProstT5/result.pt  --train_metadata newdata/S8754.csv   --test_input newdata/s669/ProstT5/result.pt   --test_metadata newdata/S669.csv -o newdata/train_test/ProstT5-11.1.csv



def Prot_t5(model_checkpoint, device):

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", trust_remote_code=True)
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print("Model loaded on:", model.device)
    return model, tokenizer


def extract_mean_representations(model, tokenizer, fasta_file, device, batch_size=16):
    mean_representations = {}
    print(f"Reading sequences from {fasta_file} FASTA file")
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    print(f'Number of sequences to process: {len(sequences)}')

    preprocessed_seqs = []
    seq_ids = []
    seq_lengths = []
    for seq in sequences:
        seq_ids.append(seq.id)
        original_seq = str(seq.seq)
        seq_lengths.append(len(original_seq))
        #Replace rare/fuzzy amino acid (uzob) with X, add space separation (adapt to prott5 input format)
        processed_seq = re.sub(r"[UZOB]", "X", original_seq)
        processed_seq = " ".join(list(processed_seq))
        preprocessed_seqs.append(processed_seq)

    with torch.no_grad():  
       
        hidden_size = model.config.d_model if hasattr(model, "config") else None
        layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=True).to(device) if hidden_size else None

        for i in tqdm(range(0, len(preprocessed_seqs), batch_size), desc="Processing sequences"):
            batch_seqs = preprocessed_seqs[i:i+batch_size]
            batch_ids = seq_ids[i:i+batch_size]
            batch_lengths = seq_lengths[i:i+batch_size]

            # tokenize
            tokens = tokenizer.batch_encode_plus(
                batch_seqs,
                return_tensors="pt",
                padding="longest",  # padding
                truncation=False
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            
            output = model(**tokens, output_hidden_states=True)
            embeddings = output.hidden_states[-1]  # (batch_size, seq_len, hidden_size)

            # LayerNorm
            if layer_norm:
                embeddings = layer_norm(embeddings)

            for j in range(len(batch_seqs)):
                seq_id = batch_ids[j]
                seq_len = batch_lengths[j]
                valid_embeddings = embeddings[j, 1:1+seq_len, :].detach().cpu()
                mean_representations[seq_id] = valid_embeddings.mean(dim=0)
    
    return mean_representations


def main():
    parser = argparse.ArgumentParser(description="Extracting ESMC representations from a FASTA file")
    parser.add_argument("-i", "--input_fasta", type=str, required=True, help="Path to the input FASTA file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing (default: 32)")
    args = parser.parse_args()

    path_input_fasta_file = args.input_fasta
    model_checkpoint = 1  
    output_file = args.output
    batch_size = args.batch_size

    base_dir = os.path.dirname(output_file)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = Prot_t5(model_checkpoint, device)

    result = extract_mean_representations(
        model, 
        tokenizer, 
        path_input_fasta_file, 
        device=device,
        batch_size=batch_size
    )
    
    torch.save(result, output_file)
    print(f'Process Finished! Results saved to {output_file}')


if __name__ == "__main__":
    main()