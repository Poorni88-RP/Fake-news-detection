export interface FactCheckCitation {
  title: string;
  url: string;
  snippet?: string;
  sourceType: "News Media" | "Fact Checker" | "Official Registry" | "Other";
  reliability: "High" | "Medium" | "Low / Partisan";
}

export interface LinguisticMarker {
  phrase: string;
  category: "Clickbait" | "Emotional Bias" | "Sensationalism" | "Vague Source" | "Logical Fallacy";
  explanation: string;
  severity: "Low" | "Medium" | "High";
  startIndex: number;
}

export interface NaiveBayesFeature {
  word: string;
  pReal: number;
  pFake: number;
  wordCount: number;
  logRatio: number; // log(pReal / pFake)
}

export interface NaiveBayesModelSim {
  classProbabilityReal: number;
  classProbabilityFake: number;
  priorReal: number;
  priorFake: number;
  criticalFeatures: NaiveBayesFeature[];
  summary: string;
}

export interface LstmGateActivation {
  sentenceIndex: number;
  textSnippet: string;
  forgetGate: number;     // 0 to 1
  inputGate: number;      // 0 to 1
  outputGate: number;     // 0 to 1
  cellStateTrend: number; // -1 (Fake bias) to +1 (Real bias)
}

export interface LstmModelSim {
  gateSequence: LstmGateActivation[];
  sequenceStabilityPercent: number; // Measure of thematic consistency
  narrativeDrift: "Consistent Narrative" | "Moderate Narrative Drifts" | "Abrupt Context Shifts";
  summary: string;
}

export interface NlpLexicalStats {
  lexicalDiversity: number;  // Type-Token ratio
  readabilityScore: number;  // Flesch etc.
  subjectivityRatio: number; // 0 (pure objective) to 1 (pure subjective)
  nounVerbRatio: number;     // linguistic structure indicator
  adjectiveDensity: number;  // sensationalism indicator
}

export interface AnalysisSummary {
  credibilityScore: number;  // 0 to 100
  label: "Highly Credible" | "Verified Real" | "Mixed Accuracy" | "Suspicious / Unverified" | "Deceptive / Fabricated";
  verdictDescription: string;
  
  // Real Fact Checking Check (Grounded)
  factCheckCitations: FactCheckCitation[];
  groundingSourcesExplanation: string;
  
  // Real Source Analysis
  sourceDomain: string;
  sourceReliabilityScore: number; // 0 to 100
  sourceBiasCategory: "Left Bias" | "Left-Center" | "Center / Neutral" | "Right-Center" | "Right Bias" | "Partisan / Conspiracy" | "Unknown / User Provided";
  sourceBiasesExplanation: string;

  // Linguistic Highlighting
  clickbaitScore: number;
  sensationalismScore: number;
  linguisticMarkers: LinguisticMarker[];

  // Interactive Model Simulation Data
  naiveBayesSim: NaiveBayesModelSim;
  lstmSim: LstmModelSim;
  lexicalStats: NlpLexicalStats;
}
