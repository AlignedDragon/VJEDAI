from pathlib import Path
import base64
import zlib
import numpy as np
import pandas as pd

def encode_depth(depth: np.ndarray) -> str:
    depth = np.asarray(depth, dtype=np.float16)
    compressed = zlib.compress(depth.tobytes(), level=9)
    return base64.b64encode(compressed).decode("utf-8")

def save_submission(rows, out_csv="../submission.csv"):
    out_csv = Path(out_csv)
    df = pd.DataFrame(rows, columns=["id", "Depths"])
    df.to_csv(out_csv, index=False)

    print(f"Saved submission to {out_csv}")
    print(f"Number of predictions: {len(df)}")

    return df