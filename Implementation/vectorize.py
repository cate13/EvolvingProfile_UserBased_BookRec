import sys
from pathlib import Path
from tqdm import tqdm 
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

from VectorMakers.EmotionVectorMaker import EmotionVectorMaker

SCRIPT_DIR = Path(__file__).resolve().parent

DESCRIPTIONS_DIR = SCRIPT_DIR.parent / "Goodreads_data" / "descriptions"

def load_txt(folder_path: Path) -> dict[int, str]:
    descriptions = {}

    for file_path in folder_path.glob("*.txt"):
        # Ensure the filename (without extension) is numeric
        if file_path.stem.isdigit():
            file_id = int(file_path.stem)
            descriptions[file_id] = file_path.read_text(encoding="utf-8")

    return descriptions


def vectorize_descriptions():
    descriptions = load_txt(DESCRIPTIONS_DIR)

    # EmoLex is boolean 
    emo_lex_maker = EmotionVectorMaker(use_intensity=False)

    # NRC-EIL is intensity 
    nrc_eil_maker = EmotionVectorMaker(use_intensity=True)

    output_path = SCRIPT_DIR.parent / "Goodreads_data" / "vectors" / "description_emotion_vectors.jsonl"

    last_written_id = "17876149"
    resuming = False

    # Use append mode ('a') so you don't overwrite existing progress
    with open(output_path, "a", encoding="utf-8") as outfile:
        for id, txt in tqdm(descriptions.items()):
            # Skip records until after the last written book_id is found
            if resuming:
                if str(id) == last_written_id:
                    resuming = False  # Start processing on the next loop iteration
                continue

            emo_lex = emo_lex_maker.getEmotionVector(txt, removeObj=True)
            nrc_eil = nrc_eil_maker.getEmotionVector(txt, removeObj=True)
            book = {
                "book_id" : id,
                "emo_lex" : emo_lex,
                "nrc_eil" : nrc_eil
            }
            outfile.write(json.dumps(book) + "\n")
        

vectorize_descriptions()