(() => {
  "use strict";

  // Mistral 5-band warm palette + Donor comparison blue
  const FLAME = ["#e10500", "#fa500f", "#ff8200", "#ffaf00", "#ffd93d"];
  const DONOR_BLUE = "#4f8bff";
  const OK_GREEN = "#3fb950";

  // Exact verified parameter scan from bolt3_b_qwen3_4b_latest.pth (step=100)
  const LAYER_SCAN_QWEN3 = [
    { l: 0, kind: "bolt3", gam: "0.9697-0.9698", kap: "0.9979", bw: "0.0140", ew: "0.6046", rms: "0.0505" },
    { l: 1, kind: "bolt3", gam: "0.9695-0.9698", kap: "0.9970", bw: "0.0724", ew: "0.8206", rms: "0.0512" },
    { l: 2, kind: "bolt3", gam: "0.9694-0.9699", kap: "0.9967", bw: "0.1441", ew: "0.9346", rms: "0.0513" },
    { l: 3, kind: "bolt3", gam: "0.9692-0.9697", kap: "0.9968", bw: "0.3211", ew: "1.0507", rms: "0.0544" },
    { l: 4, kind: "bolt3", gam: "0.9693-0.9698", kap: "0.9998", bw: "0.2409", ew: "0.9817", rms: "0.0547" },
    { l: 5, kind: "bolt3", gam: "0.9692-0.9700", kap: "0.9935", bw: "0.2339", ew: "0.9659", rms: "0.0535" },
    { l: 6, kind: "bolt3", gam: "0.9691-0.9698", kap: "0.9964", bw: "0.2549", ew: "0.9628", rms: "0.0525" },
    { l: 7, kind: "bolt3", gam: "0.9694-0.9700", kap: "1.0015", bw: "0.4674", ew: "0.9902", rms: "0.0521" },
    { l: 8, kind: "bolt3", gam: "0.9692-0.9699", kap: "1.0013", bw: "0.3684", ew: "0.9887", rms: "0.0533" },
    { l: 9, kind: "bolt3", gam: "0.9691-0.9700", kap: "1.0046", bw: "0.5123", ew: "1.0945", rms: "0.0558" },
    { l: 10, kind: "bolt3", gam: "0.9692-0.9697", kap: "1.0018", bw: "0.4706", ew: "1.0440", rms: "0.0539" },
    { l: 11, kind: "bolt3", gam: "0.9692-0.9698", kap: "1.0032", bw: "0.4422", ew: "1.0615", rms: "0.0549" },
    { l: 12, kind: "bolt3", gam: "0.9692-0.9697", kap: "1.0027", bw: "0.5108", ew: "1.0983", rms: "0.0553" },
    { l: 13, kind: "bolt3", gam: "0.9692-0.9697", kap: "1.0049", bw: "0.5121", ew: "1.0641", rms: "0.0544" },
    { l: 14, kind: "cold", gam: "0.9697-0.9697", kap: "1.0000", bw: "0.0000", ew: "0.0000", rms: "0.0500" },
    { l: 15, kind: "bolt3", gam: "0.9693-0.9699", kap: "0.9994", bw: "0.5045", ew: "0.9986", rms: "0.0522" },
    { l: 16, kind: "bolt3", gam: "0.9691-0.9697", kap: "1.0001", bw: "0.5585", ew: "1.0678", rms: "0.0542" },
    { l: 17, kind: "bolt3", gam: "0.9692-0.9698", kap: "1.0002", bw: "0.5731", ew: "1.0866", rms: "0.0536" },
    { l: 18, kind: "bolt3", gam: "0.9694-0.9698", kap: "0.9958", bw: "0.5230", ew: "1.0127", rms: "0.0523" },
    { l: 19, kind: "cold", gam: "0.9697-0.9697", kap: "1.0000", bw: "0.0000", ew: "0.0000", rms: "0.0500" },
    { l: 20, kind: "bolt3", gam: "0.9693-0.9699", kap: "1.0022", bw: "0.5372", ew: "1.0477", rms: "0.0532" },
    { l: 21, kind: "bolt3", gam: "0.9693-0.9699", kap: "1.0010", bw: "0.5340", ew: "0.9828", rms: "0.0514" },
    { l: 22, kind: "bolt3", gam: "0.9694-0.9699", kap: "1.0036", bw: "0.5540", ew: "0.9743", rms: "0.0518" },
    { l: 23, kind: "bolt3", gam: "0.9696-0.9698", kap: "1.0009", bw: "0.5389", ew: "0.9675", rms: "0.0498" },
    { l: 24, kind: "cold", gam: "0.9697-0.9697", kap: "1.0000", bw: "0.0000", ew: "0.0000", rms: "0.0500" },
    { l: 25, kind: "bolt3", gam: "0.9693-0.9701", kap: "1.0018", bw: "0.5567", ew: "0.9340", rms: "0.0502" },
    { l: 26, kind: "bolt3", gam: "0.9692-0.9702", kap: "0.9986", bw: "0.5572", ew: "0.9605", rms: "0.0538" },
    { l: 27, kind: "bolt3", gam: "0.9692-0.9700", kap: "1.0007", bw: "0.5571", ew: "0.9486", rms: "0.0533" },
    { l: 28, kind: "bolt3", gam: "0.9696-0.9699", kap: "0.9967", bw: "0.5346", ew: "0.9303", rms: "0.0501" },
    { l: 29, kind: "cold", gam: "0.9697-0.9697", kap: "1.0000", bw: "0.0000", ew: "0.0000", rms: "0.0500" },
    { l: 30, kind: "bolt3", gam: "0.9695-0.9702", kap: "1.0016", bw: "0.5502", ew: "0.9076", rms: "0.0513" },
    { l: 31, kind: "bolt3", gam: "0.9697-0.9700", kap: "0.9973", bw: "0.5470", ew: "0.9012", rms: "0.0492" },
    { l: 32, kind: "bolt3", gam: "0.9696-0.9699", kap: "0.9975", bw: "0.5115", ew: "0.8685", rms: "0.0502" },
    { l: 33, kind: "bolt3", gam: "0.9694-0.9698", kap: "0.9906", bw: "0.5454", ew: "0.8711", rms: "0.0544" },
    { l: 34, kind: "bolt3", gam: "0.9694-0.9699", kap: "1.0007", bw: "0.5259", ew: "0.7800", rms: "0.0540" },
    { l: 35, kind: "bolt3", gam: "0.9693-0.9699", kap: "1.0021", bw: "0.5154", ew: "0.8225", rms: "0.0518" }
  ];

  // Exact verified parameter scan from bolt3.1-qwen3.5-4b.pth (step=501)
  const LAYER_SCAN_QWEN35 = [
    { l: 0, kind: "gdn", gam: "decay_a 0.093..0.381", kap: "dt +1.293", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 1, kind: "gdn", gam: "decay_a 7.03..49.71", kap: "dt -3.237", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 2, kind: "gdn", gam: "decay_a 13.01..74.63", kap: "dt -3.829", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 3, kind: "bolt3", gam: "0.9953-0.9957", kap: "0.8641", bw: "2.1413", ew: "2.2604", rms: "0.0678" },
    { l: 4, kind: "gdn", gam: "decay_a 3.53..18.29", kap: "dt -3.249", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 5, kind: "gdn", gam: "decay_a 2.27..20.40", kap: "dt -2.809", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 6, kind: "gdn", gam: "decay_a 3.53..54.60", kap: "dt -3.721", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 7, kind: "bolt3", gam: "0.9957-0.9961", kap: "0.8481", bw: "2.3693", ew: "2.4289", rms: "0.1052" },
    { l: 8, kind: "gdn", gam: "decay_a 1.74..7.74", kap: "dt -3.396", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 9, kind: "gdn", gam: "decay_a 1.64..27.03", kap: "dt -3.808", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 10, kind: "gdn", gam: "decay_a 2.39..10.92", kap: "dt -4.161", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 11, kind: "cold", gam: "0.9959-0.9963", kap: "0.8890", bw: "2.3950", ew: "2.7415", rms: "0.1141" },
    { l: 12, kind: "gdn", gam: "decay_a 3.28..74.63", kap: "dt -3.890", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 13, kind: "gdn", gam: "decay_a 2.37..13.59", kap: "dt -3.540", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 14, kind: "gdn", gam: "decay_a 3.25..33.12", kap: "dt -3.742", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 15, kind: "bolt3", gam: "0.9960-0.9962", kap: "0.8907", bw: "2.4065", ew: "2.8095", rms: "0.1279" },
    { l: 16, kind: "gdn", gam: "decay_a 0.75..5.04", kap: "dt -3.668", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 17, kind: "gdn", gam: "decay_a 2.62..22.06", kap: "dt -3.523", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 18, kind: "gdn", gam: "decay_a 0.92..10.92", kap: "dt -3.328", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 19, kind: "cold", gam: "0.9959-0.9962", kap: "0.8747", bw: "2.4138", ew: "2.5697", rms: "0.1321" },
    { l: 20, kind: "gdn", gam: "decay_a 0.47..4.73", kap: "dt -2.879", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 21, kind: "gdn", gam: "decay_a 0.33..1.79", kap: "dt -1.917", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 22, kind: "gdn", gam: "decay_a 0.29..3.20", kap: "dt -0.916", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 23, kind: "bolt3", gam: "0.9959-0.9963", kap: "0.7808", bw: "2.2730", ew: "2.5461", rms: "0.1088" },
    { l: 24, kind: "gdn", gam: "decay_a 0.20..0.47", kap: "dt -1.439", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 25, kind: "gdn", gam: "decay_a 0.29..2.31", kap: "dt -0.862", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 26, kind: "gdn", gam: "decay_a 0.22..0.62", kap: "dt +0.226", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 27, kind: "cold", gam: "0.9958-0.9961", kap: "0.7268", bw: "2.2922", ew: "2.5207", rms: "0.1190" },
    { l: 28, kind: "gdn", gam: "decay_a 0.33..3.01", kap: "dt -0.069", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 29, kind: "gdn", gam: "decay_a 0.23..0.60", kap: "dt -0.173", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 30, kind: "gdn", gam: "decay_a 0.26..1.02", kap: "dt -0.976", bw: "24 GDN", ew: "native", rms: "2.05 MB" },
    { l: 31, kind: "bolt3", gam: "0.9956-0.9958", kap: "0.8439", bw: "2.3094", ew: "2.2425", rms: "0.1012" }
  ];

  const MODELS = {
    qwen35: {
      eyebrow: "24 GATEDDELTANET + 8 BOLT3ATTN (3 COLD [11,19,27]) • 0 PURE SOFTMAX • QWEN/QWEN3.5-4B",
      hfRepo: "rikkathree/bolt-qwen3.5-4b",
      hfUrl: "https://huggingface.co/rikkathree/bolt-qwen3.5-4b",
      donor: "Qwen/Qwen3.5-4B",
      ckpt: "bolt3.1-qwen3.5-4b.pth",
      stats: [
        ["75.0% / 62.5%", "4-Needle NIAH (64k / 128k)", "18/24 @64k • 15/24 @128k (4/4 single-pass @4k-32k)"],
        ["88.0% / 80.5%", "IFEval / HumanEval Parity", "Donor: IFEval 87.5% • HumanEval 80.5% • All 7 p >= 0.125"],
        ["-58.4%", "State RAM @ 64k (Flat >32k)", "873 MB vs 2,097 MB Donor (1.27x tok/s)"],
        ["1.005×", "Val PPL vs Donor (KL 0.0080)", "Top1 97.75% + TruthfulQA mc1 71.73% / mc2 34.15%"]
      ],
      layersTitle: "Inspect All 32 Layers of <code>bolt3.1-qwen3.5-4b.pth</code>",
      layersDesc: "Click any layer tile from the verified CPU parameter scan (step = 501, <code>L_mem</code> supervised). Green tiles are native preserved <code>GatedDeltaNet</code> layers (24/32), Flame tiles are compressed pure <code>O(1)</code> <code>Bolt3Attn</code> layers (5/32), and Gold tiles are Option-B Cold-Armed <code>Bolt3Attn</code> layers (<code>[11, 19, 27]</code>) with active <code>beta</code>/<code>eta</code> gates.",
      defaultLayer: 11,
      layers: LAYER_SCAN_QWEN35,
      benchTitle: "Verified Against <code>Qwen/Qwen3.5-4B</code> on RTX PRO 6000",
      benchSub: "Mistral Flame = Bolt-3.1 Option-B • Blue = Qwen/Qwen3.5-4B Donor (HF eager GDN fallback; all p >= 0.125)",
      niahHtml: "<strong>Multi-Needle NIAH, TruthfulQA &amp; Matched Controls (<code>Qwen/Qwen3.5-4B</code> Bolt-3.1):</strong> Paired single-pass 4-Needle NIAH hits <strong>4/4 (100%) at <code>4k, 16k, 32k</code></strong> and <code>2/4</code> at <code>64k</code>; multi-seed <code>64k_32k</code> reaches <strong>75.0% (18/24)</strong> and <code>128k_32k</code> reaches <strong>62.5% (15/24)</strong> (<code>d=0.10</code> evicted needles up from <code>0/8</code> to <code>2/8</code>). Beats equal-budget <code>SWA_W640+Cold</code> across <code>16k/64k</code> WikiText KL (<code>0.0660/0.1272</code> vs <code>0.0798/0.1464</code>) while still trailing full-byte <code>SWA_W898+Cold</code> (<code>0.0624/0.1137</code>). UK AISI TruthfulQA matches Donor at <strong>71.73% mc1</strong> &amp; <strong>34.15% mc2</strong> (<code>p >= 0.125</code>, <code>0</code> format errors), plus <strong>100% (4/4) Disk <code>.vsa</code> Recall</strong> (<code>0.0</code> erase residual).",
      maxY: 2250,
      maxLabel: "2.1G",
      tasks: [
        { name: "MMLU", bolt: 67.00, donor: 67.50, p: "1.000" },
        { name: "ARC-Chall.", bolt: 51.50, donor: 51.50, p: "1.000" },
        { name: "HellaSwag", bolt: 61.50, donor: 62.00, p: "1.000" },
        { name: "GSM8K", bolt: 46.50, donor: 49.50, p: "0.461" },
        { name: "HumanEval", bolt: 80.49, donor: 80.49, p: "1.000" },
        { name: "MBPP", bolt: 55.00, donor: 57.00, p: "0.125" },
        { name: "IFEval", bolt: 88.00, donor: 87.50, p: "1.000" }
      ],
      scaling: [
        { ctx: 2048, label: "2k", donorMb: 113.12, boltMb: 141.21, saved: -24.8, donorTps: 38.2, boltTps: 73.2, spd: 1.92 },
        { ctx: 8192, label: "8k", donorMb: 305.12, boltMb: 286.28, saved: 6.2, donorTps: 39.4, boltTps: 67.3, spd: 1.71 },
        { ctx: 16384, label: "16k", donorMb: 561.12, boltMb: 479.71, saved: 14.5, donorTps: 39.0, boltTps: 60.2, spd: 1.54 },
        { ctx: 32768, label: "32k", donorMb: 1073.12, boltMb: 866.57, saved: 19.2, donorTps: 37.5, boltTps: 49.4, spd: 1.32 },
        { ctx: 65536, label: "64k", donorMb: 2097.12, boltMb: 873.38, saved: 58.4, donorTps: 38.9, boltTps: 49.3, spd: 1.27 }
      ]
    },
    qwen3: {
      eyebrow: "36 / 36 BOLT3ATTN LAYERS (4 COLD [14,19,24,29]) • 0 PURE SOFTMAX • QWEN/QWEN3-4B",
      hfRepo: "rikkathree/bolt-qwen3-4b",
      hfUrl: "https://huggingface.co/rikkathree/bolt-qwen3-4b",
      donor: "Qwen/Qwen3-4B",
      ckpt: "bolt3_b_qwen3_4b_latest.pth",
      stats: [
        ["90.00%", "GSM8K Math (N=200)", "Donor Qwen3-4B: 88.00% (McNemar p=0.523)"],
        ["1.49×", "Decode Speed @ 64k", "44.0 vs 29.6 tok/s (128-tok chunks)"],
        ["-87.1%", "Live Runtime RAM @ 64k", "1,186 MB vs 9,216 MB Donor KV"],
        ["100.0%", "1-Needle NIAH (2k–16k)", "+ 4/4 Disk .vsa Recall (0.0 erase)"]
      ],
      layersTitle: "Inspect All 36 Layers of <code>bolt3_b_qwen3_4b_latest.pth</code>",
      layersDesc: "Click any layer tile from the verified CPU parameter scan (step = 100). Notice how the 32 compressed layers adapted <code>beta_proj</code> and <code>eta_proj</code> gates while the 4 sparse Cold-Armed layers stayed untouched until sequences exceed 32k overflow.",
      defaultLayer: 14,
      layers: LAYER_SCAN_QWEN3,
      benchTitle: "Verified Against <code>Qwen/Qwen3-4B</code> on RTX PRO 6000",
      benchSub: "Mistral Flame = Bolt-3 Option-B • Blue = Qwen/Qwen3-4B Donor (all p >= 0.05)",
      niahHtml: "<strong>Needle-in-a-Haystack (WikiText-103) Diagnostic Transparency:</strong> On single-passkey retrieval (<strong>1-Needle NIAH</strong>), Qwen3-4B Option-B scored <strong>100.0% (4/4)</strong> across <code>2k, 4k, 8k, and 16k</code>. On simultaneous 4-record associative recall (<strong>4-Needle NIAH</strong>), accuracy dropped past 4k (`75% @ 4k -> 0% @ 32k`) under full 36-layer retrofit, motivating our <strong>Qwen3.5-4B</strong> hybrid release which achieves <strong>4/4 @ 32k</strong>.",
      maxY: 9500,
      maxLabel: "9.2G",
      tasks: [
        { name: "MMLU", bolt: 68.50, donor: 69.50, p: "0.790" },
        { name: "ARC-Chall.", bolt: 56.00, donor: 54.00, p: "0.455" },
        { name: "HellaSwag", bolt: 55.50, donor: 52.00, p: "0.118" },
        { name: "GSM8K", bolt: 90.00, donor: 88.00, p: "0.523" },
        { name: "HumanEval", bolt: 76.22, donor: 80.49, p: "0.230" },
        { name: "MBPP", bolt: 53.50, donor: 55.50, p: "0.481" },
        { name: "IFEval", bolt: 87.50, donor: 86.50, p: "0.774" }
      ],
      scaling: [
        { ctx: 2048, label: "2k", donorMb: 288.00, boltMb: 214.82, saved: 25.4, donorTps: 59.4, boltTps: 70.4, spd: 1.19 },
        { ctx: 8192, label: "8k", donorMb: 1152.00, boltMb: 408.06, saved: 64.6, donorTps: 58.2, boltTps: 63.5, spd: 1.09 },
        { ctx: 16384, label: "16k", donorMb: 2304.00, boltMb: 665.72, saved: 71.1, donorTps: 59.3, boltTps: 55.5, spd: 0.94 },
        { ctx: 32768, label: "32k", donorMb: 4608.00, boltMb: 1181.03, saved: 74.4, donorTps: 46.7, boltTps: 44.1, spd: 0.94 },
        { ctx: 65536, label: "64k", donorMb: 9216.00, boltMb: 1186.11, saved: 87.1, donorTps: 29.6, boltTps: 44.0, spd: 1.49 }
      ]
    }
  };

  let activeModelKey = "qwen35";
  let currentModel = MODELS.qwen35;
  let SCALING = currentModel.scaling;

  // --- Decorative Mistral Mosaics ---
  const brandMosaic = document.getElementById("brand-mosaic");
  const mosaicPattern = [
    1, 0, 0, 0, 1,
    1, 1, 0, 1, 1,
    1, 2, 3, 2, 1,
    1, 0, 4, 0, 1,
    0, 0, 5, 0, 0
  ];
  if (brandMosaic) {
    mosaicPattern.forEach((v) => {
      const i = document.createElement("i");
      i.style.background = v === 0 ? "transparent" : FLAME[v - 1];
      brandMosaic.appendChild(i);
    });
  }

  ["pixel-band-1", "pixel-band-2"].forEach((id, idx) => {
    const el = document.getElementById(id);
    if (!el) return;
    for (let i = 0; i < 24; i++) {
      const s = document.createElement("span");
      const band = (i + idx * 3) % 8;
      if (band < 5) {
        s.style.background = FLAME[band];
        s.style.opacity = "0.78";
      } else {
        s.style.background = "#14141c";
      }
      el.appendChild(s);
    }
  });

  // --- Hero Mistral Pixel Horizon Canvas ---
  const heroCanvas = document.getElementById("hero-canvas");
  let heroMouse = { x: 0.6, y: 0.4 };
  window.addEventListener("pointermove", (e) => {
    heroMouse.x = e.clientX / Math.max(1, window.innerWidth);
    heroMouse.y = e.clientY / Math.max(1, window.innerHeight);
  }, { passive: true });

  function fitCanvas(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(300, Math.round(rect.width));
    const h = Math.max(200, Math.round(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function renderHero(t) {
    if (!heroCanvas) return;
    const { ctx, w, h } = fitCanvas(heroCanvas);
    ctx.clearRect(0, 0, w, h);

    const tile = w < 640 ? 18 : 22;
    const cols = Math.ceil(w / tile);
    const rows = Math.ceil(h / tile);
    const time = t * 0.0009;
    const sy = Math.min(1.5, window.scrollY / Math.max(1, window.innerHeight));

    // Geometric Mistral pixel wave sweeping right-to-left across horizon
    for (let c = 0; c < cols; c++) {
      const nx = c / cols;
      const dx = nx - heroMouse.x;
      const baseH =
        0.54 +
        0.14 * Math.sin(nx * 5.2 - time * 1.1 + sy * 2.4) +
        0.09 * Math.cos(nx * 11.0 + time * 0.7) +
        Math.exp(-dx * dx * 18) * 0.11;

      const horizonRow = Math.floor(baseH * rows);
      for (let r = 0; r < rows; r++) {
        const distFromWave = horizonRow - r;
        if (distFromWave >= -1 && distFromWave <= 6) {
          const band = Math.max(0, Math.min(4, Math.floor(((distFromWave + 1) / 7) * 5)));
          const alpha = (0.14 + 0.42 * nx) * (1 - Math.abs(distFromWave - 2) * 0.16) * (1 - sy * 0.55);
          if (alpha <= 0.02) continue;
          ctx.fillStyle = FLAME[4 - band];
          ctx.globalAlpha = Math.max(0.03, Math.min(0.82, alpha));
          ctx.fillRect(c * tile + 1, r * tile + 1, tile - 3, tile - 3);
        } else if ((c + r) % 7 === 0 && r > horizonRow && r < horizonRow + 5) {
          ctx.fillStyle = "#fa500f";
          ctx.globalAlpha = 0.06 * (1 - sy * 0.5);
          ctx.fillRect(c * tile + 1, r * tile + 1, tile - 3, tile - 3);
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  // --- Scroll Progress + Reveal Animations ---
  const progressBar = document.getElementById("scroll-progress");
  const revealEls = Array.from(document.querySelectorAll("[data-reveal]"));
  const barFills = Array.from(document.querySelectorAll(".bar-fill"));

  const revealObs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) e.target.classList.add("revealed");
    });
  }, { threshold: 0.12 });
  revealEls.forEach((el) => revealObs.observe(el));

  // --- Scrollytelling Architecture Visualizer ---
  const archCanvas = document.getElementById("arch-canvas");
  const stepCards = Array.from(document.querySelectorAll(".step-card"));
  const stepPills = Array.from(document.querySelectorAll(".step-pill"));
  const hudStage = document.getElementById("hud-stage");
  const hudGeom = document.getElementById("hud-geom");
  const hudMem = document.getElementById("hud-mem");
  const modeOptB = document.getElementById("mode-optb");
  const modeFull = document.getElementById("mode-full");

  let activeStep = 0;
  let smoothStep = 0;
  let useOptionB = true;

  const STAGE_META = [
    { name: "01 / O(N) Softmax Bottleneck", geom: "Full Softmax KV • Linear RAM Growth", memOptB: "Donor @64k: 2.1G–9.2G", memFull: "Donor @64k: 2.1G–9.2G" },
    { name: "02 / Sink + 4C Unified Ring", geom: "n_sinks=4 • 4C=512 Ring • CUDA Graph", memOptB: "Unified in-place buf_k", memFull: "Unified in-place buf_k" },
    { name: "03 / Tri-Partition Attractors", geom: "S=128 (48 HH + 48 Land + 32 Proto)", memOptB: "-31–36% KL vs TrueNoAtt", memFull: "-31–36% KL vs TrueNoAtt" },
    { name: "04 / Gershgorin M_res + L_mem", geom: "Matrix [HKV, HD, HD] float32", memOptB: "Trained via L_mem loss", memFull: "Trained via L_mem loss" },
    { name: "05 / Option-B Cold Layers", geom: "Sparse Cold [11,19,27] • max=256", memOptB: "873 MB @64k (-58.4%)", memFull: "~150 MB flat all contexts" },
    { name: "06 / Disk .vsa + Exact Forget", geom: "Cosine router (33.6 ms) • Linear delta", memOptB: "4/4 recall • 0.0 erase err", memFull: "4/4 recall • 0.0 erase err" }
  ];

  function setStep(idx) {
    activeStep = Math.max(0, Math.min(5, idx));
    stepCards.forEach((c, i) => c.classList.toggle("is-active", i === activeStep));
    stepPills.forEach((p, i) => p.classList.toggle("active", i === activeStep));
    const meta = STAGE_META[activeStep];
    if (hudStage) hudStage.textContent = meta.name;
    if (hudGeom) hudGeom.textContent = meta.geom;
    if (hudMem) hudMem.textContent = useOptionB ? meta.memOptB : meta.memFull;
  }

  stepPills.forEach((p) => {
    p.addEventListener("click", () => {
      const idx = Number(p.dataset.step);
      setStep(idx);
      const card = stepCards[idx];
      if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });

  stepCards.forEach((c) => {
    c.addEventListener("click", () => setStep(Number(c.dataset.step)));
  });

  if (modeOptB && modeFull) {
    modeOptB.addEventListener("click", () => {
      useOptionB = true;
      modeOptB.classList.add("active");
      modeFull.classList.remove("active");
      setStep(activeStep);
    });
    modeFull.addEventListener("click", () => {
      useOptionB = false;
      modeFull.classList.add("active");
      modeOptB.classList.remove("active");
      setStep(activeStep);
    });
  }

  function computeScrollStep() {
    const mid = window.innerHeight * 0.45;
    let best = 0;
    let bestDist = Infinity;
    stepCards.forEach((card, i) => {
      const r = card.getBoundingClientRect();
      const center = r.top + r.height * 0.5;
      const dist = Math.abs(center - mid);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    });
    if (best !== activeStep) setStep(best);

    const docH = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
    if (progressBar) progressBar.style.width = `${Math.min(100, (window.scrollY / docH) * 100)}%`;
  }

  window.addEventListener("scroll", computeScrollStep, { passive: true });

  function drawBox(ctx, x, y, w, h, stroke, fill, label, sub) {
    ctx.fillStyle = fill;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.25;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    if (label) {
      ctx.fillStyle = "#f3efe6";
      ctx.font = "600 12px 'JetBrains Mono', monospace";
      ctx.fillText(label, x + 10, y + 20);
    }
    if (sub) {
      ctx.fillStyle = "#9b958b";
      ctx.font = "400 10.5px 'JetBrains Mono', monospace";
      ctx.fillText(sub, x + 10, y + 36);
    }
  }

  function renderArch(t) {
    if (!archCanvas) return;
    const { ctx, w, h } = fitCanvas(archCanvas);
    smoothStep += (activeStep - smoothStep) * 0.11;
    const phase = t * 0.0022;

    // Subtle dark editorial grid
    ctx.fillStyle = "#09090d";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(255,255,255,0.035)";
    ctx.lineWidth = 1;
    for (let gx = 0; gx < w; gx += 28) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
    }
    for (let gy = 0; gy < h; gy += 28) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }

    const pad = 22;
    const stage = Math.round(smoothStep);

    // Header inside canvas
    ctx.fillStyle = FLAME[3];
    ctx.font = "700 11px 'JetBrains Mono', monospace";
    const titles = [
      "UNBOUNDED O(N) SOFTMAX KV CACHE STACKING",
      "UNIFIED BUFFER + C=128 DOUBLE-BUFFERED CHUNK STREAM",
      "COSINE MERGE (>=0.94 K, >=0.88 V) -> S=128 ATTRACTORS",
      "GERSHGORIN NORMALIZED DELTA WRITES -> M_res [8 x 128 x 128]",
      useOptionB ? "OPTION-B: 32 PURE O(1) + 4 SPARSE COLD [14,19,24,29]" : "PURE O(1) BOLT-3: ALL 36 LAYERS BOUNDED (~150 MB FLAT)",
      "BOLT3MEMORYBANK: .VSA CARTRIDGE ROUTE + EXACT FORGET()"
    ];
    ctx.fillText(titles[stage], pad, 24);

    if (stage === 0) {
      // Unbounded KV growth pixel columns
      const nCols = Math.min(34, Math.floor((w - pad * 2) / 16));
      const grow = Math.floor(((Math.sin(phase * 0.8) + 1) * 0.5) * nCols);
      for (let i = 0; i < nCols; i++) {
        const colX = pad + i * 16;
        const blocks = 3 + Math.floor((i / nCols) * 14);
        const lit = i <= grow;
        for (let b = 0; b < blocks; b++) {
          const by = h - 54 - b * 14;
          if (by < 48) break;
          ctx.fillStyle = lit ? DONOR_BLUE : "rgba(79, 139, 255, 0.14)";
          ctx.fillRect(colX, by, 12, 11);
        }
      }
      ctx.fillStyle = "#f85149";
      ctx.font = "600 12px 'JetBrains Mono', monospace";
      ctx.fillText("Linear KV explosion across all 36 layers -> 9,216 MB @ 64k (29.6 tok/s)", pad, h - 20);
    } else if (stage === 1) {
      // Unified CUDA Graph buffer strip: [sink | ak_eff | half0 | half1 | cold?]
      const topY = 58;
      const boxH = 74;
      const totW = w - pad * 2;
      const segs = [
        { lbl: "SINK", sub: "4 tok", frac: 0.09, col: FLAME[0] },
        { lbl: "AK_EFF", sub: "S=128 tri-part", frac: 0.27, col: FLAME[1] },
        { lbl: "LOC_RING", sub: "4C=512 exact", frac: 0.43, col: FLAME[2] },
        { lbl: useOptionB ? "COLD" : "NO COLD", sub: useOptionB ? "sparse+hit" : "0 slots", frac: 0.21, col: useOptionB ? FLAME[4] : "#2b2b36" }
      ];
      let cx = pad;
      const curSlot = Math.floor(((phase * 4) % 16));
      segs.forEach((sg, idx) => {
        const sw = Math.floor(totW * sg.frac) - 4;
        drawBox(ctx, cx, topY, sw, boxH, sg.col, "rgba(255,255,255,0.03)", sg.lbl, sg.sub);
        // Animated token cursor inside LOC_RING
        if (idx === 2) {
          for (let k = 0; k < 16; k++) {
            const kw = (sw - 16) / 16;
            ctx.fillStyle = k <= curSlot ? FLAME[3] : "rgba(255,175,0,0.14)";
            ctx.fillRect(cx + 8 + k * kw, topY + boxH - 18, Math.max(2, kw - 2), 10);
          }
        }
        cx += sw + 5;
      });

      // Lower incoming stream -> index_copy_ pulse
      const streamY = topY + 126;
      drawBox(ctx, pad, streamY, totW, 118, "rgba(255,130,0,0.35)", "#0d0d14",
        "_Bolt3GraphRunner Zero-Cat Steady Step (B=1 or B<=64)",
        "buf_k.index_copy_(2, slot_buf_t, k_p)  |  Q @ buf_k^T -> mask_buf -> Softmax -> util_acc");
      for (let i = 0; i < 20; i++) {
        const pulse = Math.sin(phase * 2.5 - i * 0.45) * 0.5 + 0.5;
        ctx.fillStyle = FLAME[Math.min(4, Math.floor(pulse * 5))];
        ctx.globalAlpha = 0.25 + 0.7 * pulse;
        ctx.fillRect(pad + 14 + i * ((totW - 28) / 20), streamY + 58, 12, 34 * pulse + 8);
      }
      ctx.globalAlpha = 1;
    } else if (stage === 2) {
      // Diversity Attractors: tokens flowing into attractor mosaic, merging or evicting
      const leftW = Math.floor((w - pad * 2) * 0.34);
      const rightX = pad + leftW + 46;
      const rightW = w - pad - rightX;

      drawBox(ctx, pad, 56, leftW, h - 96, FLAME[2], "#0e0e16", "EVICTING PREV CHUNK", "128 RoPE-raw keys");
      drawBox(ctx, rightX, 56, rightW, h - 96, FLAME[3], "#0e0e16", "ATTRACTOR BANK (S=128)", "ak, av, mass, pm, util");

      // Draw pixel attractors on right
      const cols = 8;
      const rows = 8;
      const cell = Math.min(26, Math.floor((rightW - 28) / cols));
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c;
          const actPulse = Math.sin(phase * 1.4 + i * 0.7) * 0.5 + 0.5;
          const massHot = (i % 7 === 0);
          ctx.fillStyle = massHot ? FLAME[4] : FLAME[Math.min(3, Math.floor(actPulse * 3.8))];
          ctx.globalAlpha = massHot ? 0.95 : 0.25 + 0.55 * actPulse;
          ctx.fillRect(rightX + 14 + c * cell, 110 + r * cell, cell - 4, cell - 4);
        }
      }
      ctx.globalAlpha = 1;

      // Animate merging / evicting token blocks across gap
      for (let k = 0; k < 5; k++) {
        const prog = ((phase * 0.55 + k * 0.2) % 1);
        const px = pad + leftW - 14 + prog * (rightX - (pad + leftW) + 28);
        const py = 120 + k * 42 + Math.sin(prog * Math.PI) * -14;
        const merged = k % 2 === 0;
        ctx.fillStyle = merged ? FLAME[4] : FLAME[0];
        ctx.fillRect(px, py, 12, 12);
        if (k === 0) {
          ctx.fillStyle = "#f3efe6";
          ctx.font = "400 10px 'JetBrains Mono', monospace";
          ctx.fillText("merge >=0.94 -> mass+=1", pad + 12, h - 54);
          ctx.fillText("evict score_j -> M_res", pad + 12, h - 38);
        }
      }
    } else if (stage === 3) {
      // Gershgorin M_res square matrix grid
      const gridDim = 12;
      const matSize = Math.min(h - 110, Math.floor((w - pad * 2) * 0.48));
      const cell = Math.floor(matSize / gridDim);
      const mx = w - pad - gridDim * cell - 10;
      const my = 60;

      drawBox(ctx, pad, 60, w - pad * 2 - matSize - 32, h - 100, FLAME[1], "#0d0d14",
        "EVICTED SURPLUS TOKENS", "beta_proj -> Gershgorin step");
      ctx.fillStyle = "#9b958b";
      ctx.font = "400 11px 'JetBrains Mono', monospace";
      ctx.fillText("w_ev = sigmoid(beta) * delta", pad + 12, 120);
      ctx.fillText("gersh = clamp(|K*K^T| * w, 1)", pad + 12, 140);
      ctx.fillText("gamma_delta ~ 0.9697", pad + 12, 160);
      ctx.fillText("tanh(eta_proj) * res_norm(y)", pad + 12, 180);

      // Animated matrix writes
      ctx.strokeStyle = FLAME[2];
      ctx.strokeRect(mx - 6, my - 6, gridDim * cell + 12, gridDim * cell + 12);
      for (let r = 0; r < gridDim; r++) {
        for (let c = 0; c < gridDim; c++) {
          const val = Math.sin(phase * 1.6 + r * 0.6 - c * 0.4) * 0.5 + 0.5;
          const diag = r === c;
          ctx.fillStyle = diag ? FLAME[4] : FLAME[Math.min(3, Math.floor(val * 4))];
          ctx.globalAlpha = diag ? 0.9 : 0.16 + val * 0.68;
          ctx.fillRect(mx + c * cell + 1, my + r * cell + 1, cell - 3, cell - 3);
        }
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = FLAME[4];
      ctx.font = "600 11px 'JetBrains Mono', monospace";
      ctx.fillText("M_res [8 x 128 x 128] float32", mx, my + gridDim * cell + 22);
    } else if (stage === 4) {
      // Option-B 36 layer stack vs Pure O(1)
      const cols = 18;
      const rows = 2;
      const tileW = Math.floor((w - pad * 2 - 20) / cols);
      const tileH = 52;
      for (let i = 0; i < 36; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const tx = pad + col * tileW;
        const ty = 58 + row * (tileH + 10);
        const isCold = useOptionB && (i === 14 || i === 19 || i === 24 || i === 29);
        ctx.fillStyle = isCold ? FLAME[4] : "rgba(225, 5, 0, 0.25)";
        ctx.fillRect(tx + 2, ty, tileW - 4, tileH);
        ctx.strokeStyle = isCold ? FLAME[4] : FLAME[1];
        ctx.strokeRect(tx + 2, ty, tileW - 4, tileH);
        ctx.fillStyle = isCold ? "#09090c" : "#f3efe6";
        ctx.font = "700 9.5px 'JetBrains Mono', monospace";
        ctx.fillText(`L${String(i).padStart(2, "0")}`, tx + 5, ty + 16);
        ctx.fillText(isCold ? "COLD" : "O(1)", tx + 5, ty + 32);
      }

      const barY = 200;
      const maxBarW = w - pad * 2;
      if (useOptionB) {
        drawBox(ctx, pad, barY, maxBarW, h - barY - 32, FLAME[3], "#0d0d14",
          "FIFO COLD BUCKETS (0 -> 16 -> 32 -> 64 -> 128 -> 256) + RoPE INVERSION PAST 32k",
          "32,768 tokens cap -> oldest block pops via apply_rope_pos(ok_p, -ok_pos) into ak + M_res");
        const fillFrac = Math.min(1, ((phase * 0.25) % 1.25));
        ctx.fillStyle = FLAME[3];
        ctx.fillRect(pad + 14, barY + 62, (maxBarW - 28) * Math.min(0.75, fillFrac), 16);
        if (fillFrac > 0.75) {
          ctx.fillStyle = FLAME[0];
          ctx.fillRect(pad + 14 + (maxBarW - 28) * 0.75, barY + 62, (maxBarW - 28) * (fillFrac - 0.75), 16);
        }
        ctx.strokeStyle = "rgba(255,255,255,0.2)";
        ctx.strokeRect(pad + 14, barY + 62, maxBarW - 28, 16);
      } else {
        drawBox(ctx, pad, barY, maxBarW, h - barY - 32, FLAME[1], "#0d0d14",
          "PURE O(1) MODE (bolt3_cold_layers = [])",
          "Zero KV retention across all 36 layers -> constant ~150 MB state from 2k to infinity");
        ctx.fillStyle = FLAME[1];
        ctx.fillRect(pad + 14, barY + 62, (maxBarW - 28) * 0.15, 16);
        ctx.strokeStyle = "rgba(255,255,255,0.2)";
        ctx.strokeRect(pad + 14, barY + 62, maxBarW - 28, 16);
      }
    } else {
      // Stage 5: .vsa cartridge routing + forget
      const carts = ["orion.vsa", "hemlock.vsa", "vanguard.vsa", "chronos.vsa"];
      const hitIdx = Math.floor((phase * 0.45) % 4);
      const cw = Math.floor((w - pad * 2 - 36) / 4);
      carts.forEach((name, idx) => {
        const cx = pad + idx * (cw + 12);
        const active = idx === hitIdx;
        drawBox(ctx, cx, 60, cw, 72, active ? OK_GREEN : "rgba(255,255,255,0.14)",
          active ? "rgba(63, 185, 80, 0.14)" : "#0d0d14", name, active ? "COSINE MATCH" : "disk dormant");
      });

      const boxY = 175;
      drawBox(ctx, pad, boxY, w - pad * 2, h - boxY - 28, FLAME[2], "#0d0d14",
        "INJECT DELTA TRACE -> ANSWER -> EXACT FORGET()",
        "memorize: delta = st - gam^n * pre  |  forget: st -= gam^k * delta -> residual = 0.00e+00");
      const subPhase = (phase * 0.6) % 2;
      const barW = (w - pad * 2 - 40) * (subPhase < 1 ? subPhase : 2 - subPhase);
      ctx.fillStyle = subPhase < 1 ? OK_GREEN : FLAME[0];
      ctx.fillRect(pad + 20, boxY + 68, Math.max(4, barW), 14);
    }
  }

  // --- Interactive Model Switcher + Layer Grid + 7-Task Bar Chart ---
  const layerGrid = document.getElementById("layer-grid");
  const taskBars = document.getElementById("task-bars");

  function selectLayer(li) {
    const layers = currentModel.layers;
    const info = layers[li] || layers[0];
    document.querySelectorAll(".layer-tile").forEach((el, i) => el.classList.toggle("selected", i === info.l));
    const badge = info.kind === "cold" ? "[Cold-Armed]" : (info.kind === "gdn" ? "[GatedDeltaNet Native]" : "[Pure Bolt-3]");
    document.getElementById("ld-idx").textContent = `L${String(info.l).padStart(2, "0")} ${badge}`;
    document.getElementById("ld-gam-k").textContent = info.kind === "gdn" ? "Decay a Range" : "Gamma Delta Range";
    document.getElementById("ld-kap-k").textContent = info.kind === "gdn" ? "dt_bias Mean" : "Kappa Median";
    document.getElementById("ld-beta-k").textContent = info.kind === "gdn" ? "Backbone" : "Beta Gate |W|";
    document.getElementById("ld-eta-k").textContent = info.kind === "gdn" ? "Mode / Status" : "Eta Gate |W|";
    document.getElementById("ld-gam").textContent = info.gam;
    document.getElementById("ld-kap").textContent = info.kap;
    document.getElementById("ld-beta").textContent = info.bw;
    document.getElementById("ld-eta").textContent = info.ew;
  }

  function renderLayers() {
    if (!layerGrid) return;
    layerGrid.innerHTML = "";
    currentModel.layers.forEach((row) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `layer-tile ${row.kind}`;
      const tag = row.kind === "cold" ? "COLD" : (row.kind === "gdn" ? "GDN" : `η:${row.ew}`);
      btn.innerHTML = `L${String(row.l).padStart(2, "0")}<span class="l-tag">${tag}</span>`;
      btn.addEventListener("click", () => selectLayer(row.l));
      layerGrid.appendChild(btn);
    });
    selectLayer(currentModel.defaultLayer);
  }

  function renderTaskBars() {
    if (!taskBars) return;
    taskBars.innerHTML = "";
    currentModel.tasks.forEach((t) => {
      const row = document.createElement("div");
      row.className = "task-row";
      row.innerHTML = `
        <div class="task-name">${t.name}</div>
        <div class="bar-pair">
          <div class="bar-track"><div class="bar-fill bolt" data-w="${t.bolt}%" style="width:${t.bolt}%"></div></div>
          <div class="bar-track"><div class="bar-fill donor" data-w="${t.donor}%" style="width:${t.donor}%"></div></div>
        </div>
        <div class="task-scores">
          <div style="color:var(--m-4)">B3: ${t.bolt.toFixed(1)}%</div>
          <div style="color:var(--donor)">HF: ${t.donor.toFixed(1)}% <span style="color:var(--text-faint)">p=${t.p}</span></div>
        </div>
      `;
      taskBars.appendChild(row);
    });
  }

  function switchModel(key) {
    activeModelKey = key;
    currentModel = MODELS[key];
    SCALING = currentModel.scaling;

    const b35 = document.getElementById("model-qwen35");
    const b3 = document.getElementById("model-qwen3");
    if (b35 && b3) {
      b35.classList.toggle("active", key === "qwen35");
      b3.classList.toggle("active", key === "qwen3");
    }
    const c35 = document.getElementById("card-qwen35");
    const c3 = document.getElementById("card-qwen3");
    if (c35 && c3) {
      c35.classList.toggle("is-active", key === "qwen35");
      c3.classList.toggle("is-active", key === "qwen3");
    }
    const ey = document.getElementById("hero-eyebrow");
    if (ey) ey.textContent = currentModel.eyebrow;
    const hfNav = document.getElementById("nav-hf-link");
    if (hfNav) {
      hfNav.href = currentModel.hfUrl;
      hfNav.textContent = key === "qwen35" ? "HF :: Qwen3.5-4B" : "HF :: Qwen3-4B";
    }
    currentModel.stats.forEach((s, i) => {
      const v = document.getElementById(`stat-v${i}`);
      const l = document.getElementById(`stat-l${i}`);
      const sub = document.getElementById(`stat-s${i}`);
      if (v) v.textContent = s[0];
      if (l) l.textContent = s[1];
      if (sub) sub.textContent = s[2];
    });
    const lt = document.getElementById("layers-title");
    const ld = document.getElementById("layers-desc");
    if (lt) lt.innerHTML = currentModel.layersTitle;
    if (ld) ld.innerHTML = currentModel.layersDesc;
    const bt = document.getElementById("bench-title");
    const bs = document.getElementById("bench-sub");
    if (bt) bt.innerHTML = currentModel.benchTitle;
    if (bs) bs.textContent = currentModel.benchSub;
    const nc = document.getElementById("niah-callout");
    if (nc) nc.innerHTML = currentModel.niahHtml;
    const pre = document.getElementById("cli-serve-pre");
    if (pre) {
      pre.textContent = `python -m bolt3 scan --ckpt ${currentModel.ckpt}\npython -m bolt3 benchmark --ckpt ${currentModel.ckpt} --donor ${currentModel.donor}\npython -m bolt3 serve --ckpt ${currentModel.ckpt} --donor ${currentModel.donor} --port 8000`;
    }

    renderLayers();
    renderTaskBars();
    updateScaling(Number(ctxSlider ? ctxSlider.value : 4));
  }

  const btnQ35 = document.getElementById("model-qwen35");
  const btnQ3 = document.getElementById("model-qwen3");
  if (btnQ35 && btnQ3) {
    btnQ35.addEventListener("click", () => switchModel("qwen35"));
    btnQ3.addEventListener("click", () => switchModel("qwen3"));
  }
  const cBtnQ35 = document.getElementById("card-btn-qwen35");
  const cBtnQ3 = document.getElementById("card-btn-qwen3");
  if (cBtnQ35) {
    cBtnQ35.addEventListener("click", () => {
      switchModel("qwen35");
      document.getElementById("layers")?.scrollIntoView({ behavior: "smooth" });
    });
  }
  if (cBtnQ3) {
    cBtnQ3.addEventListener("click", () => {
      switchModel("qwen3");
      document.getElementById("layers")?.scrollIntoView({ behavior: "smooth" });
    });
  }

  // --- Context Scaling SVG + Scrubber ---
  const ctxSlider = document.getElementById("ctx-slider");
  const ctxReadout = document.getElementById("ctx-readout");
  const scRam = document.getElementById("sc-ram");
  const scSpd = document.getElementById("sc-spd");
  const scalingSvg = document.getElementById("scaling-svg");

  function drawScalingSvg(activeIdx) {
    if (!scalingSvg) return;
    const W = 480, H = 220, P = 42, PR = 18, PT = 18, PB = 34;
    const pw = W - P - PR, ph = H - PT - PB;
    const xPos = (i) => P + (i / (SCALING.length - 1)) * pw;
    const yRam = (mb) => PT + ph - (mb / currentModel.maxY) * ph;

    const donorPts = SCALING.map((d, i) => `${xPos(i).toFixed(1)},${yRam(d.donorMb).toFixed(1)}`).join(" ");
    const boltPts = SCALING.map((d, i) => `${xPos(i).toFixed(1)},${yRam(d.boltMb).toFixed(1)}`).join(" ");
    const ax = xPos(activeIdx);
    const active = SCALING[activeIdx];
    const maxD = SCALING[SCALING.length - 1].donorMb;

    scalingSvg.innerHTML = `
      <line x1="${P}" y1="${PT + ph}" x2="${W - PR}" y2="${PT + ph}" stroke="rgba(255,255,255,0.18)" />
      <line x1="${P}" y1="${PT}" x2="${P}" y2="${PT + ph}" stroke="rgba(255,255,255,0.18)" />
      <text x="${P - 6}" y="${PT + 6}" fill="#67625b" font-size="9" font-family="JetBrains Mono" text-anchor="end">${currentModel.maxLabel}</text>
      <text x="${P - 6}" y="${PT + ph}" fill="#67625b" font-size="9" font-family="JetBrains Mono" text-anchor="end">0</text>
      <polyline fill="none" stroke="${DONOR_BLUE}" stroke-width="2.2" stroke-dasharray="5,4" points="${donorPts}" />
      <polyline fill="none" stroke="${FLAME[2]}" stroke-width="3" points="${boltPts}" />
      <line x1="${ax}" y1="${PT}" x2="${ax}" y2="${PT + ph}" stroke="rgba(255,175,0,0.45)" stroke-dasharray="3,3" />
      ${SCALING.map((d, i) => `
        <circle cx="${xPos(i)}" cy="${yRam(d.donorMb)}" r="${i === activeIdx ? 5 : 3}" fill="${DONOR_BLUE}" />
        <circle cx="${xPos(i)}" cy="${yRam(d.boltMb)}" r="${i === activeIdx ? 5.5 : 3.5}" fill="${FLAME[3]}" />
        <text x="${xPos(i)}" y="${H - 12}" fill="${i === activeIdx ? "#ffd93d" : "#9b958b"}" font-size="10" font-family="JetBrains Mono" text-anchor="middle">${d.label}</text>
      `).join("")}
      <text x="${W - PR}" y="${PT + 12}" fill="${DONOR_BLUE}" font-size="10" font-family="JetBrains Mono" text-anchor="end">Donor KV (${Math.round(maxD).toLocaleString()} MB @64k)</text>
      <text x="${W - PR}" y="${yRam(active.boltMb) - 10}" fill="#ffaf00" font-size="10" font-family="JetBrains Mono" text-anchor="end">Bolt-3 Option-B (${active.boltMb} MB)</text>
    `;
  }

  function updateScaling(idx) {
    const d = SCALING[idx];
    const diffLabel = d.saved >= 0 ? `-${d.saved}%` : `+${Math.abs(d.saved)}%`;
    if (ctxReadout) ctxReadout.textContent = `${d.ctx.toLocaleString()} tok`;
    if (scRam) scRam.textContent = `${d.boltMb.toFixed(2)} vs ${d.donorMb.toFixed(2)} MB (${diffLabel})`;
    if (scSpd) scSpd.textContent = `${d.boltTps.toFixed(1)} vs ${d.donorTps.toFixed(1)} tok/s (${d.spd.toFixed(2)}x)`;
    drawScalingSvg(idx);
  }

  if (ctxSlider) {
    ctxSlider.addEventListener("input", () => updateScaling(Number(ctxSlider.value)));
  }

  // --- Changelog Modal ---
  const clBtn = document.getElementById("changelog-btn");
  const clModal = document.getElementById("changelog-modal");
  const clClose = document.getElementById("cl-close");
  const clBackdrop = document.getElementById("cl-backdrop");
  const openChangelog = () => {
    if (!clModal) return;
    clModal.hidden = false;
    clClose?.focus();
  };
  const closeChangelog = () => {
    if (!clModal) return;
    clModal.hidden = true;
    clBtn?.focus();
  };
  clBtn?.addEventListener("click", openChangelog);
  clClose?.addEventListener("click", closeChangelog);
  clBackdrop?.addEventListener("click", closeChangelog);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && clModal && !clModal.hidden) {
      closeChangelog();
    }
  });

  // --- Animation Loop ---
  function loop(t) {
    renderHero(t);
    renderArch(t);
    requestAnimationFrame(loop);
  }
  switchModel("qwen35");
  setStep(0);
  computeScrollStep();
  requestAnimationFrame(loop);
})();
