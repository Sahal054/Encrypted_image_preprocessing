# Encrypted Fingerprint Identification Using Fully Homomorphic Encryption

Identify fingerprints without ever exposing the raw biometric image to the server.
The client encrypts a fingerprint image with a secret key that never leaves the
device; the server applies ridge-enhancement and feature-extraction filters
**directly on ciphertext** using Zama's Concrete-ML; the client decrypts the
result locally.  The server learns nothing about the fingerprint at any point.

---

## System Design

### High-level architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT (browser)                            │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────────┐  │
│  │  Gradio   │──▶│  FHEClient   │──▶│  Private key + eval key   │  │
│  │   UI      │   │  encrypt()   │   │  (never leaves client)    │  │
│  │           │◀──│  decrypt()   │◀──│                            │  │
│  └──────────┘   └──────┬───────┘   └────────────────────────────┘  │
│                         │  ciphertext + eval_key                    │
└─────────────────────────┼───────────────────────────────────────────┘
                          │  HTTPS
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SERVER (FastAPI)                               │
│                                                                     │
│  ┌──────────────┐   ┌────────────────────────────────────────────┐  │
│  │  FHEServer    │──▶│  Compiled FHE circuits (one per filter)  │  │
│  │  .run()       │   │  filters/<name>/deployment/server.zip     │  │
│  └──────────────┘   └────────────────────────────────────────────┘  │
│                                                                     │
│  The server has NO secret key. It can only evaluate the circuit     │
│  on encrypted data and return an encrypted result.                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Data flow (step by step)

| Step | Actor | Operation | Data in transit |
|------|-------|-----------|-----------------|
| 1 | Client | User uploads fingerprint image | — |
| 2 | Client | `FHEClient.keygen()` — generate secret + evaluation keys | — |
| 3 | Client | `FHEClient.encrypt_serialize(image)` | — |
| 4 | Client → Server | POST `/send_input` | encrypted_image ‖ evaluation_key |
| 5 | Server | `FHEServer.run(encrypted_image, eval_key)` | — |
| 6 | Server → Client | POST `/get_output` | encrypted_output |
| 7 | Client | `FHEClient.deserialize_decrypt_post_process(output)` | — |
| 8 | Client | Display original ‖ decrypted result | — |

At no point does the server possess the secret key or any plaintext pixel.

### Fingerprint processing pipeline in FHE

The available filters form a layered pipeline that mirrors classical
fingerprint identification — except every operation runs on encrypted
integers:

```
                 ┌─────────────┐
  Raw fingerprint │ black & white │  Grayscale conversion (PAL/NTSC weights)
                 └──────┬──────┘
                        │
                 ┌──────┴──────┐
                   blur          Noise reduction (3×3 box filter ÷ 9)
                 └──────┬──────┘
                        │
            ┌───────────┴───────────┐
            │                       │
     ┌──────┴──────┐        ┌──────┴──────┐
       sharpen        ridge detection    Local contrast & ridge boost
     └──────┬──────┘        └──────┬──────┘
            │                       │
            └───────────┬───────────┘
                        │
            ┌───────────┴───────────────────┐
            │           │                   │
     ┌──────┴──────┐ ┌─┴──────────┐ ┌──────┴──────┐
     sobel horiz.  │ sobel vert. │  laplacian     │
     └──────┬──────┘ └─────┬─────┘  └──────┬──────┘
            │              │               │
            └──────────────┴───────────────┘
                           │
                  Directional ridge / edge
                  feature maps (encrypted)
                           │
                           ▼
                  Fingerprint feature vector
                  (ready for encrypted matching)
```

Each block is a single integer-coefficient convolution compiled into an
independent FHE circuit.  They can be chained: the client decrypts the
output of one filter, re-encrypts, and sends it through the next — or a
single filter can be used standalone for quick testing.

> **Why convolutions only?**  Fully homomorphic encryption over the
> torus (TFHE) supports arbitrary combinational circuits via table
> lookups, but the most efficient operations are integer additions and
> multiplications.  Convolutions are exactly that: weighted sums of
> encrypted pixels with public integer kernels — no floating point, no
> branching, no data-dependent memory access.

### Component map

| File | Responsibility |
|------|---------------|
| `app.py` | Gradio UI, client-side keygen / encrypt / decrypt, orchestrates HTTP calls to the server |
| `server.py` | FastAPI backend — receives ciphertext + eval key, runs compiled FHE circuit, returns encrypted result |
| `client_server_interface.py` | `FHEServer` / `FHEDev` / `FHEClient` — thin wrappers around Concrete's `Server`, `Client`, and artifact I/O |
| `filters.py` | `TorchConv` model + `Filter` wrapper — defines kernels, compiles to FHE circuits, post-processing |
| `common.py` | Centralised constants: paths, `AVAILABLE_FILTERS`, `INPUT_SHAPE`, example images |
| `generate_dev_files.py` | One-time script: compiles every filter and writes `server.zip` / `client.zip` under `filters/<name>/deployment/` |

### Security guarantees

| Property | How it is achieved |
|----------|--------------------|
| **Confidentiality of input** | The image is encrypted client-side under the user's secret key before any network transmission. |
| **Confidentiality of output** | The server returns only the encrypted result; decryption requires the secret key that never left the client. |
| **Server learns nothing** | The evaluation key enables computation but **not** decryption.  This is the core property of FHE. |
| **Wrong-key test** | The demo decrypts the server's response with an independent key to produce random noise — proving that only the correct secret key recovers the true image. |
| **No plaintext on server** | At no point does `server.py` materialise a plaintext pixel.  All operations are on `fhe.Value` objects. |

### Limitations & future work

* **Image size** — FHE ciphertexts grow with the number of encrypted
  integers.  The demo uses 100×100×3 = 30 000 encrypted values, which
  keeps key-gen and execution times manageable.  Larger images are
  possible but slower.

* **Single-filter per request** — Each filter is compiled as an
  independent circuit.  A production pipeline could compose multiple
  convolutions into one circuit (reducing round-trips) at the cost of a
  larger p-error budget and longer compilation.

* **Fingerprint matching** — The current system extracts encrypted
  feature maps.  Encrypted matching (e.g., computing a dot-product or
  Hamming distance between the extracted feature vector and a stored
  template) is **directly implementable** in FHE — it requires only
  additions and multiplications — and is planned as the next milestone.

---

## Repository layout

```
app.py                          Gradio UI + client-side FHE workflow
server.py                       FastAPI backend — runs encrypted filters
client_server_interface.py      FHEClient / FHEServer / FHEDev wrappers
common.py                       Shared config, paths, filter list
filters.py                      TorchConv kernels + Filter compile/post-process
generate_dev_files.py           Compile & save FHE circuits for all filters
filters/                        Compiled deployment artifacts (server.zip, client.zip)
  ├── black and white/deployment/
  ├── blur/deployment/
  ├── sharpen/deployment/
  ├── ridge detection/deployment/
  ├── fingerprint enhance/deployment/
  ├── sobel horizontal/deployment/
  ├── sobel vertical/deployment/
  └── laplacian/deployment/
input_examples/                 Sample fingerprint images for the demo
requirements.txt                Python dependencies
```

---

## Run locally

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip3 install --upgrade pip wheel setuptools
pip3 install -r requirements.txt

# 3. (First time only) Compile FHE circuits for all filters
python3 generate_dev_files.py

# 4. Launch the application
python3 app.py
```

`app.py` automatically starts the FastAPI backend on port 8000 and then
opens the Gradio interface.  Open the URL printed in the terminal.

> **Tip:** Step 3 can take several minutes per filter.  The compiled
> artifacts are cached under `filters/`; you only need to re-run it when
> you add or modify a filter.

---

## Adding a new filter

1. **Define the kernel** in `filters.py` — add an `elif` branch in
   `Filter.__init__` with an integer-valued 1D or 2D kernel.
2. **Register the name** in `AVAILABLE_FILTERS` inside `common.py`.
3. **Recompile** all circuits: `python3 generate_dev_files.py`.
4. **Relaunch** the app: `python3 app.py`.

---

## Notes

* Deployment artifacts under `filters/*/deployment/` must be regenerated
  whenever `filters.py`, `common.py`, or the `concrete-ml` version
  changes.  Mismatched artifacts will cause deserialization errors.

* The `concrete-ml==1.9.0` pin is intentional.  Newer major versions may
  restructure the public API (`Compiler`, `NumpyModule`, etc.).  Upgrade
  with care and re-run `generate_dev_files.py`.
```

---
