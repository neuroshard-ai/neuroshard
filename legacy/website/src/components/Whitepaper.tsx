import { Download, FileText, BookOpen } from 'lucide-react';

export const Whitepaper = () => {
  // Public PDF URL - directly from GitHub
  const pdfUrl = 'https://github.com/neuroshard-ai/neuroshard/raw/main/docs/whitepaper/neuroshard_whitepaper.pdf';
  
  // Google Docs viewer for embedding PDFs (works better than direct iframe)
  const viewerUrl = `https://docs.google.com/viewer?url=${encodeURIComponent(pdfUrl)}&embedded=true`;
  
  const handleDownload = () => {
    window.open(pdfUrl, '_blank');
  };

  return (
    <div className="min-h-screen bg-neutral-950 pt-20 sm:pt-24 pb-12 px-4">
      <div className="container mx-auto max-w-5xl">
        {/* Header Section */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-accent/10">
                <BookOpen className="w-6 h-6 text-accent" />
              </div>
              <h1 className="text-3xl font-bold text-white font-display">Technical Whitepaper</h1>
            </div>
            <p className="text-neutral-400">
              Complete technical documentation of the NeuroShard protocol
            </p>
          </div>
          
          <button
            onClick={handleDownload}
            className="flex items-center gap-2 px-6 py-3 bg-accent hover:bg-accent/90 text-neutral-950 font-bold transition-all"
          >
            <Download className="w-5 h-5" />
            Download PDF
          </button>
        </div>

        {/* Access Badge */}
        <div className="mb-6 flex items-center gap-2 text-sm">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-accent/10 border border-accent/30">
            <FileText className="w-4 h-4 text-accent" />
            <span className="text-accent">Members Only</span>
          </div>
        </div>

        {/* PDF Viewer */}
        <div className="bg-neutral-900 border border-neutral-800 overflow-hidden">
          <div className="bg-neutral-800/50 px-4 py-3 border-b border-neutral-700 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-neutral-400" />
              <span className="text-neutral-300 text-sm font-medium">NeuroShard_Whitepaper.pdf</span>
            </div>
            <span className="text-xs text-neutral-500">Registered Members</span>
          </div>
          <div className="h-[75vh]">
            <iframe
              src={viewerUrl}
              className="w-full h-full border-0"
              title="NeuroShard Whitepaper"
              allow="autoplay"
            />
          </div>
        </div>

        {/* Footer Note */}
        <div className="mt-6 text-center text-neutral-500 text-sm">
          <p>
            This whitepaper is available to registered NeuroShard members.
            <br />
            <a 
              href="https://github.com/neuroshard-ai/neuroshard" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-accent hover:text-accent underline"
            >
              View source code on GitHub
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};
