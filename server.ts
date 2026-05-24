import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Helper function to lazily initialize the Gemini API client safely
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is not defined in Secrets.");
    }
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// 1. API Route for Fake News Analysis
app.post("/api/analyze", async (req, res) => {
  try {
    const { articleText, title, sourceUrl } = req.body;

    if (!articleText || typeof articleText !== "string" || articleText.trim().length === 0) {
      return res.status(400).json({ error: "Article text is required for analysis." });
    }

    let ai;
    try {
      ai = getGeminiClient();
    } catch (credentialError: any) {
      return res.status(401).json({
        error: "Gemini API key is missing. Please add your GEMINI_API_KEY inside the Settings > Secrets menu.",
        details: credentialError.message,
      });
    }

    // Prepare prompt
    const cleanText = articleText.substring(0, 10000); // Guard token length limit
    const itemTitle = title ? `Title: ${title.trim()}\n` : "";
    const itemUrl = sourceUrl ? `Source URL: ${sourceUrl.trim()}\n` : "";

    const userInstructions = `
You are an expert computational linguist, journalism ethics investigator, and fact-checker.
Analyze this news article for credibility, objectivity, factuality, and source reliability.
Evaluate how standard classification algorithms (like Naive Bayes, recurrent LSTM sequences, and lexical NLP feature analyzers) understand this specific article content.

ARTICLE DETAILS:
${itemTitle}${itemUrl}Content Summary/Full text:
${cleanText}

Use your built-in Google Search Tool to look up relevant claims, headlines, domains, associated fact-checking evaluations (from Politifact, Snopes, FactCheck.org, Reuters, etc.), and overall verify the current historical real-world context of this information as of May 2026.

Produce EXACTLY a JSON string conforming to the following structure:
{
  "credibilityScore": number (0 to 100. High means highly credible, verified real news),
  "label": "Highly Credible" | "Verified Real" | "Mixed Accuracy" | "Suspicious / Unverified" | "Deceptive / Fabricated",
  "verdictDescription": "A concise summary explaining the credibility verdict based on evidence and source reputation",
  "factCheckCitations": [
    {
      "title": "Title of the matching news, fact-check, or official page",
      "url": "full URL of the source page",
      "snippet": "Brief supporting sentence or correction context",
      "sourceType": "News Media" | "Fact Checker" | "Official Registry" | "Other",
      "reliability": "High" | "Medium" | "Low / Partisan"
    }
  ],
  "groundingSourcesExplanation": "An explanation of the web consensus, matching reports, or debunking articles found",
  "sourceDomain": "The primary domain publishing this news or 'Unknown/User Submission'",
  "sourceReliabilityScore": number (0 to 100 of the source's overall historical reputation),
  "sourceBiasCategory": "Left Bias" | "Left-Center" | "Center / Neutral" | "Right-Center" | "Right Bias" | "Partisan / Conspiracy" | "Unknown / User Provided",
  "sourceBiasesExplanation": "Factual description of the publishing outlet's historical orientation, funding, or reputation",
  "clickbaitScore": number (0 to 100 of headline/hook clickbait style),
  "sensationalismScore": number (0 to 100 of syntactic alarmism, outrage, or emotional bias),
  "linguisticMarkers": [
    {
      "phrase": "Specific word or short sequence of words from the article that is highly sensationalized, Clickbait, or misleading",
      "category": "Clickbait" | "Emotional Bias" | "Sensationalism" | "Vague Source" | "Logical Fallacy",
      "explanation": "Why this phrase indicates bias or deception",
      "severity": "Low" | "Medium" | "High",
      "startIndex": number (estimated 0-based index of where the phrase begins in the analyzed text)
    }
  ],
  "naiveBayesSim": {
    "classProbabilityReal": number (0.0 to 1.0 based on joint P(Word|Class) products of the terms in the article),
    "classProbabilityFake": number (0.0 to 1.0 based on joint P(Word|Class) products),
    "priorReal": number (average baseline real probability, e.g., 0.5),
    "priorFake": number (average baseline fake probability, e.g., 0.5),
    "criticalFeatures": [
      {
        "word": "The lowercase keyword extracted",
        "pReal": number (conditional probability P(word|Real), e.g., 0.045),
        "pFake": number (conditional probability P(word|Fake), e.g., 0.082),
        "wordCount": number (how many times it occurred in the text),
        "logRatio": number (log ratio of P(word|Real)/P(word|Fake))
      }
    ],
    "summary": "Explain how a Bag-of-Words Naive Bayes Classifier processes this specific vocab distribution based on word conditional ratios."
  },
  "lstmSim": {
    "gateSequence": [
      {
        "sentenceIndex": number (index 0, 1, 2...),
        "textSnippet": "Short direct quote of this sentence or sequence from the text",
        "forgetGate": number (0.0 to 1.0 activation. Visualizes cell state reset. Abrupt context switches, bias injection, or contradiction pushes forget gate close to 1 to wipe history. Smooth narrative retains logic with lower forget gate values),
        "inputGate": number (0.0 to 1.0 activation. The standard amount of new information added to memory),
        "outputGate": number (0.0 to 1.0 activation. The level of state filtered through for the immediate cell output),
        "cellStateTrend": number (-1.0 to 1.0, where -1 represents strong deceptive context and +1 representing verifiable factual state accumulation)
      }
    ],
    "sequenceStabilityPercent": number (0 to 100 measure of narrative cohesion across sentences),
    "narrativeDrift": "Consistent Narrative" | "Moderate Narrative Drifts" | "Abrupt Context Shifts",
    "summary": "Explain how the sequential state updates of the LSTM capture temporal narratives, tracking context shifts or outrage-pattern progression better than isolated words."
  },
  "lexicalStats": {
    "lexicalDiversity": number (Type-Token Ratio, typically 0.3 to 0.8),
    "readabilityScore": number (Flesch readability estimate, 0 to 100. Low means complex academic language, high means simple catchy standard),
    "subjectivityRatio": number (0.0 pure facts to 1.0 pure extreme opinions),
    "nounVerbRatio": number (structural style check average, e.g., 1.4),
    "adjectiveDensity": number (ratio of adjectives to other tokens, typical 0.05 to 0.25. High density means heavily descriptive/hype style)
  }
}

CRITICAL: Return ONLY valid, un-decorated JSON. Do not wrap it in markdown block tags except regular response body. Double check all keys and values to avoid parsing issues.
`;

    // Query Gemini
    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: userInstructions,
      config: {
        tools: [{ googleSearch: {} }],
        responseMimeType: "application/json",
      },
    });

    const responseText = response.text;
    if (!responseText) {
      throw new Error("No response content generated from Gemini core.");
    }

    // Try parsing the response JSON
    let analysisData: any;
    try {
      analysisData = JSON.parse(responseText.trim());
    } catch (parseError: any) {
      console.error("JSON Parse fail. Raw response:", responseText);
      // Fallback clean extraction
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        analysisData = JSON.parse(jsonMatch[0]);
      } else {
        throw new Error("Failed to receive structured analysis from Gemini. Response was: " + responseText.substring(0, 200));
      }
    }

    // Extract real-time search grounding metadata to enrich citations
    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    if (groundingChunks && Array.isArray(groundingChunks)) {
      const parsedCitations = groundingChunks
        .filter((chunk: any) => chunk.web && chunk.web.uri)
        .map((chunk: any, index: number) => ({
          title: chunk.web.title || `Grounding Source #${index + 1}`,
          url: chunk.web.uri,
          snippet: "Referenced via Google Search Grounding for real-time verification.",
          sourceType: "News Media" as const,
          reliability: "High" as const,
        }));

      // Merge verified links if no cites were populated dynamically
      if (!analysisData.factCheckCitations || analysisData.factCheckCitations.length === 0) {
        analysisData.factCheckCitations = parsedCitations;
      } else {
        // Dedup and append
        const urls = new Set(analysisData.factCheckCitations.map((c: any) => c.url));
        parsedCitations.forEach((pc: any) => {
          if (!urls.has(pc.url)) {
            analysisData.factCheckCitations.push(pc);
          }
        });
      }
    }

    // Return final structured payload
    return res.json(analysisData);

  } catch (error: any) {
    console.error("Analysis Exception:", error);
    return res.status(500).json({
      error: "An internal error occurred during news article evaluation.",
      details: error.message || String(error),
    });
  }
});

// Serve frontend assets
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Fake News Detector Express server run on port ${PORT}`);
  });
}

startServer();
