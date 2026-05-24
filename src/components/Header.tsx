import { ShieldAlert, Cpu, Sparkles } from "lucide-react";

export function Header() {
  return (
    <header id="app-header" className="border-b border-slate-800 bg-slate-950/70 backdrop-blur-md sticky top-0 z-50 py-4 px-6">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Branding & Logo */}
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-cyan-500/10 rounded-xl border border-cyan-500/20 shadow-[0_0_15px_rgba(6,182,212,0.15)] animate-pulse">
            <ShieldAlert className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-[10px] tracking-widest text-cyan-400 font-bold uppercase">
                Veritas Core Engine v3.1
              </span>
              <span className="bg-slate-800 text-slate-300 font-mono text-[9px] px-1.5 py-0.5 rounded border border-slate-700">
                ACTIVE
              </span>
            </div>
            <h1 className="text-xl md:text-2xl font-bold font-sans tracking-tight text-white flex items-center gap-2">
              Fake News Detector
            </h1>
          </div>
        </div>

        {/* Technical Sub-banner */}
        <div className="flex items-center gap-6 text-slate-400 text-xs font-mono">
          <div className="flex items-center gap-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800">
            <Cpu className="w-3.5 h-3.5 text-slate-500" />
            <span>ALGORITHMS:</span>
            <span className="text-slate-200">NLP • Naive Bayes • LSTM • Transformers</span>
          </div>
          <div className="hidden md:flex items-center gap-1.5 text-cyan-400 bg-cyan-950/35 px-2.5 py-1 rounded-full border border-cyan-900/35">
            <Sparkles className="w-3 h-3" />
            <span>Search Grounded AI</span>
          </div>
        </div>
      </div>
    </header>
  );
}
