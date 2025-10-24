// server.js (top)
import express from "express";
import multer from "multer";
import { OpenAI } from "openai";
import { z } from "zod";

import helmet from "helmet";
import cors from "cors";
import rateLimit from "express-rate-limit";

const app = express();
app.set("trust proxy", 1);

// tighten later to your app’s domain(s)
app.use(cors({ origin: true }));
app.use(helmet());

// Basic per-IP rate limit (tune for prod)
app.use(rateLimit({
  windowMs: 60_000, // 1 minute
  limit: 60,        // 60 req/min
  standardHeaders: true,
  legacyHeaders: false
}));

// Simple client token auth (MVP). Improve later with Firebase/JWT.
const CLIENT_TOKEN = process.env.CLIENT_TOKEN || "";
app.use((req, res, next) => {
  if (!CLIENT_TOKEN) return next(); // allow if no token set
  if (req.header("X-Client-Token") === CLIENT_TOKEN) return next();
  return res.status(401).json({ error: "unauthorized" });
});

// Health check
app.get("/health", (req, res) => res.send("ok"));

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Multer with limits (avoid huge uploads)
const upload = multer({
  limits: {
    files: 1,
    fileSize: 3 * 1024 * 1024 // 3MB
  }
});

// Upstream timeout helper
const withTimeout = (p, ms = 60000) =>
  Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error("upstream-timeout")), ms))]);

// ===== existing zod schema remains the same =====
const ItemsSchema = z.object({
  items: z.array(
    z.object({
      label: z.string(),
      confidence: z.number(),
      canonical: z.string(),
    })
  ),
});

// Multipart food recognizer (your existing endpoint)
app.post("/recognize", upload.single("image"), async (req, res) => {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({ error: "missing-openai-key" });
    }
    if (!req.file) {
      return res.status(400).json({ error: "missing-image" });
    }

    const base64 = req.file.buffer.toString("base64");

    const input = [
      {
        role: "system",
        content:
          "You are a FOOD-ONLY visual tagger. Output 3–5 edible foods actually visible. " +
          "If the picture is not of food, return an empty list. " +
          "NEVER output vague terms like: snack, food, meal, dish, appetizer, plate, junk food. " +
          "Prefer concrete foods (e.g., spaghetti, penne pasta, orange, steak, broccoli). " +
          "Use 'canonical' as a normalized food for kcal lookup (e.g., spaghetti→pasta, penne→pasta, fries→french fries).",
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text:
              "Return JSON only: items=[{label, confidence, canonical}]. If uncertain, still return best guesses; " +
              "if no EDIBLE items are visible, return items:[].",
          },
          {
            type: "input_image",
            image_url: `data:image/jpeg;base64,${base64}`,
          },
        ],
      },
    ];

    const response = await withTimeout(
      openai.responses.create({
        model: "gpt-4o", // or gpt-4o-mini for cost
        input,
        text: {
          format: {
            type: "json_schema",
            name: "food_labels",
            schema: {
              type: "object",
              additionalProperties: false,
              required: ["items"],
              properties: {
                items: {
                  type: "array",
                  minItems: 0,
                  maxItems: 5,
                  items: {
                    type: "object",
                    additionalProperties: false,
                    required: ["label", "confidence", "canonical"],
                    properties: {
                      label: { type: "string" },
                      confidence: { type: "number" },
                      canonical: { type: "string" },
                    },
                  },
                },
              },
            },
          },
        },
      }),
      60000
    );

    const textOut = response.output_text;
    if (!textOut) {
      return res.status(502).json({ error: "no-output-text" });
    }

    let data;
    try {
      data = JSON.parse(textOut);
    } catch (e) {
      return res.status(502).json({ error: "parse-failed", text: textOut.slice(0, 400) });
    }

    let parsed;
    try {
      parsed = ItemsSchema.parse(data);
    } catch (e) {
      return res.status(502).json({ error: "schema-failed" });
    }

    const banned = new Set(["snack", "food", "meal", "dish", "appetizer", "plate", "junk food"]);
    const itemsRaw = (parsed.items || [])
      .map((it) => ({
        label: String(it.label || "").trim().toLowerCase(),
        canonical: String(it.canonical || "").trim().toLowerCase(),
        confidence: Math.max(0, Math.min(1, Number(it.confidence) || 0)),
      }))
      .filter((it) => it.label.length > 0 && !banned.has(it.label))
      .map((it) => {
        const l = it.label;
        let canon = it.canonical || l;
        if (/(spaghetti|penne|fusilli|farfalle|macaroni|rigatoni|tagliatelle|linguine|fettuccine|pasta|noodle)/.test(l)) canon = "pasta";
        if (/(fries|french\s*fries)/.test(l)) canon = "french fries";
        if (/(orange|mandarin|tangerine)/.test(l)) canon = "orange";
        return { ...it, canonical: canon };
      })
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5);

    return res.json(itemsRaw);
  } catch (e) {
    if (e.message === "upstream-timeout") return res.status(504).json({ error: "upstream-timeout" });
    if (e.code === "LIMIT_FILE_SIZE") return res.status(413).json({ error: "file-too-large" });
    console.error("[ERROR]", e?.status, e?.code, e?.message);
    return res.status(500).json({ error: "vision-failed" });
  }
});

// (Optional) also expose a JSON endpoint so the app can send URLs or text (no multipart)
// app.post("/v1/vision", async (req, res) => { ... });

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log("Vision backend listening on", PORT));
