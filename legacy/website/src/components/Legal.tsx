import { useState } from 'react';
import { motion } from 'framer-motion';
import { Coins, Brain, AlertTriangle, Scale, FileText, Lock } from 'lucide-react';
import { SEO } from './SEO';

type Section = 'terms' | 'token' | 'llm' | 'privacy' | 'risks';

export const Legal = () => {
  const [activeSection, setActiveSection] = useState<Section>('terms');

  const sections = [
    { id: 'terms' as Section, label: 'Terms of Service', icon: FileText },
    { id: 'token' as Section, label: 'Token Disclaimer', icon: Coins },
    { id: 'llm' as Section, label: 'AI Disclaimer', icon: Brain },
    { id: 'privacy' as Section, label: 'Privacy Policy', icon: Lock },
    { id: 'risks' as Section, label: 'Risk Disclosure', icon: AlertTriangle },
  ];

  return (
    <>
      <SEO title="Legal Information" description="Terms of Service, Privacy Policy, and Risk Disclosures for NeuroShard." />
      <div className="min-h-screen bg-neutral-950 pt-20 sm:pt-24 pb-16">
      <div className="container mx-auto px-4 max-w-6xl">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-neutral-800/50 border border-neutral-700 mb-6">
            <Scale className="w-4 h-4 text-accent" />
            <span className="text-neutral-300 text-sm">Legal Information</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4 font-display">
            Terms & <span className="text-accent">Disclaimers</span>
          </h1>
          <p className="text-neutral-400 max-w-2xl mx-auto">
            Please read these terms carefully before using NeuroShard. By using our services, 
            you agree to be bound by these terms.
          </p>
          <p className="text-neutral-500 text-sm mt-4">
            Last Updated: {new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
          </p>
        </motion.div>

        {/* Important Notice Banner */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-amber-500/10 border border-amber-500/30 p-6 mb-8"
        >
          <div className="flex items-start gap-4">
            <AlertTriangle className="w-6 h-6 text-amber-400 flex-shrink-0 mt-1" />
            <div>
              <h3 className="text-amber-300 font-bold text-lg mb-2 font-display">Important Notice</h3>
              <p className="text-amber-200/80 text-sm">
                NeuroShard is an experimental decentralized AI project. NEURO tokens are utility tokens 
                with no guaranteed value. The AI model (NeuroLLM) is trained by a decentralized network 
                and may produce inaccurate, incomplete, or harmful outputs. Use at your own risk. 
                This is not financial, investment, or legal advice.
              </p>
            </div>
          </div>
        </motion.div>

        <div className="flex flex-col lg:flex-row gap-8">
          {/* Sidebar Navigation */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:w-64 flex-shrink-0"
          >
            <div className="bg-neutral-900 border border-neutral-800 p-4 sticky top-24">
              <h3 className="text-white font-bold mb-4 px-2 font-display">Sections</h3>
              <nav className="space-y-1">
                {sections.map((section) => (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 text-left transition-all ${
                      activeSection === section.id
                        ? 'bg-accent/10 text-accent border border-accent/30'
                        : 'text-neutral-400 hover:text-white hover:bg-neutral-800'
                    }`}
                  >
                    <section.icon className="w-4 h-4" />
                    <span className="text-sm font-medium">{section.label}</span>
                  </button>
                ))}
              </nav>
            </div>
          </motion.div>

          {/* Content */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex-1"
          >
            <div className="bg-neutral-900 border border-neutral-800 p-8">
              {activeSection === 'terms' && <TermsOfService />}
              {activeSection === 'token' && <TokenDisclaimer />}
              {activeSection === 'llm' && <LLMDisclaimer />}
              {activeSection === 'privacy' && <PrivacyPolicy />}
              {activeSection === 'risks' && <RiskDisclosure />}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
    </>
  );
};

const SectionTitle = ({ icon: Icon, title }: { icon: any; title: string }) => (
  <div className="flex items-center gap-3 mb-6 pb-4 border-b border-neutral-700">
    <div className="p-2 bg-accent/10">
      <Icon className="w-5 h-5 text-accent" />
    </div>
    <h2 className="text-2xl font-bold text-white font-display">{title}</h2>
  </div>
);

const SubSection = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <div className="mb-6">
    <h3 className="text-lg font-semibold text-white mb-3 font-display">{title}</h3>
    <div className="text-neutral-300 text-sm leading-relaxed space-y-2">{children}</div>
  </div>
);

const TermsOfService = () => (
  <div>
    <SectionTitle icon={FileText} title="Terms of Service" />
    
    <SubSection title="1. Acceptance of Terms">
      <p>
        By accessing or using NeuroShard ("the Service"), including the website, software, 
        NEURO tokens, and NeuroLLM AI model, you agree to be bound by these Terms of Service. 
        If you do not agree to these terms, do not use the Service.
      </p>
    </SubSection>

    <SubSection title="2. Eligibility">
      <p>
        You must be at least 18 years old to use this Service. By using the Service, you 
        represent and warrant that you are at least 18 years of age and have the legal 
        capacity to enter into these terms.
      </p>
      <p>
        You are responsible for ensuring that your use of the Service complies with all 
        laws, rules, and regulations applicable to you in your jurisdiction.
      </p>
    </SubSection>

    <SubSection title="3. Nature of Service">
      <p>
        NeuroShard is an experimental, decentralized AI training network. Key characteristics:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>The network is operated by distributed, independent nodes</li>
        <li>There is no central authority or single point of control</li>
        <li>The AI model (NeuroLLM) is trained collectively by network participants</li>
        <li>NEURO tokens are utility tokens used within the network</li>
        <li>The Service is provided "as is" without warranties of any kind</li>
      </ul>
    </SubSection>

    <SubSection title="4. User Responsibilities">
      <p>You agree to:</p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Provide accurate information when creating an account</li>
        <li>Maintain the security of your wallet and recovery phrase</li>
        <li>Not use the Service for any illegal or unauthorized purpose</li>
        <li>Not attempt to manipulate, attack, or exploit the network</li>
        <li>Not submit malicious data or poisoned gradients to the training process</li>
        <li>Comply with all applicable laws and regulations</li>
      </ul>
    </SubSection>

    <SubSection title="5. Wallet Security">
      <p>
        <strong className="text-white">You are solely responsible for your wallet security.</strong> 
        {' '}Your 12-word recovery phrase is the only way to access your wallet. NeuroShard cannot 
        recover lost phrases or reverse transactions. Never share your recovery phrase with anyone.
      </p>
    </SubSection>

    <SubSection title="6. Intellectual Property">
      <p>
        NeuroShard software is released under the Apache License 2.0. NeuroLLM model weights 
        are collectively owned by the network participants. You retain rights to any 
        data you contribute, subject to the training process.
      </p>
    </SubSection>

    <SubSection title="7. Limitation of Liability">
      <p>
        TO THE MAXIMUM EXTENT PERMITTED BY LAW, NEUROSHARD AND ITS CONTRIBUTORS SHALL NOT 
        BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, 
        INCLUDING BUT NOT LIMITED TO LOSS OF PROFITS, DATA, TOKENS, OR OTHER INTANGIBLE LOSSES.
      </p>
    </SubSection>

    <SubSection title="8. Modifications">
      <p>
        We reserve the right to modify these terms at any time. Continued use of the 
        Service after changes constitutes acceptance of the new terms.
      </p>
    </SubSection>

    <SubSection title="9. Governing Law">
      <p>
        These terms shall be governed by and construed in accordance with applicable laws, 
        without regard to conflict of law principles.
      </p>
    </SubSection>
  </div>
);

const TokenDisclaimer = () => (
  <div>
    <SectionTitle icon={Coins} title="NEURO Token Disclaimer" />
    
    <div className="bg-red-500/10 border border-red-500/30 p-4 mb-6">
      <p className="text-red-300 font-semibold text-sm">
        ⚠️ NEURO IS A UTILITY TOKEN. THIS IS NOT INVESTMENT ADVICE. DO NOT PURCHASE OR 
        ACQUIRE NEURO TOKENS WITH THE EXPECTATION OF PROFIT.
      </p>
    </div>

    <SubSection title="1. Token Nature">
      <p>
        NEURO is a <strong className="text-white">utility token</strong> designed exclusively 
        for use within the NeuroShard network. It serves the following purposes:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li><strong>Reward Currency:</strong> Earned by contributing compute and data</li>
        <li><strong>Payment Currency:</strong> Spent to access NeuroLLM inference</li>
        <li><strong>Staking:</strong> Staked to earn reward multipliers and become a validator</li>
        <li><strong>Governance:</strong> Used for NeuroDAO voting (future)</li>
      </ul>
    </SubSection>

    <SubSection title="2. No Investment Value">
      <p>
        NEURO tokens are <strong className="text-white">NOT</strong>:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Securities or investment instruments</li>
        <li>Shares, equity, or ownership in any company</li>
        <li>A store of value or speculative asset</li>
        <li>Legal tender or currency</li>
        <li>Backed by any physical asset, government, or company</li>
      </ul>
      <p className="mt-3">
        The value of NEURO tokens may fluctuate significantly and <strong className="text-white">
        may become worthless</strong>. Do not acquire NEURO tokens if you cannot afford to 
        lose the entire value.
      </p>
    </SubSection>

    <SubSection title="3. Regulatory Uncertainty">
      <p>
        The regulatory status of utility tokens and decentralized networks varies by 
        jurisdiction and may change. NEURO tokens may be:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Restricted or prohibited in your jurisdiction</li>
        <li>Subject to future regulation that limits their use</li>
        <li>Classified differently by different regulatory bodies</li>
      </ul>
      <p className="mt-3">
        <strong className="text-white">You are responsible</strong> for determining whether 
        acquiring or using NEURO tokens is legal in your jurisdiction.
      </p>
    </SubSection>

    <SubSection title="4. Tax Implications">
      <p>
        Earning, trading, or using NEURO tokens may have tax implications in your 
        jurisdiction. <strong className="text-white">You are solely responsible</strong> for 
        determining and paying any applicable taxes.
      </p>
    </SubSection>

    <SubSection title="5. No Guarantees">
      <p>We make no guarantees regarding:</p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>The future value of NEURO tokens</li>
        <li>The continued operation of the network</li>
        <li>The ability to exchange NEURO for other currencies</li>
        <li>Any specific functionality or utility</li>
      </ul>
    </SubSection>

    <SubSection title="6. No Refunds">
      <p>
        NEURO tokens are <strong className="text-white">non-refundable</strong>. Once earned 
        or transferred, transactions cannot be reversed. Lost tokens cannot be recovered.
      </p>
    </SubSection>

    <SubSection title="7. Fair Launch">
      <p>
        NeuroShard implements a <strong className="text-white">zero pre-mine</strong> policy:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>No tokens are pre-allocated to founders or investors</li>
        <li>All NEURO must be earned through Proof of Neural Work</li>
        <li>There is no ICO, IEO, or token sale</li>
        <li>Every token is traceable to a specific verified contribution</li>
      </ul>
    </SubSection>
  </div>
);

const LLMDisclaimer = () => (
  <div>
    <SectionTitle icon={Brain} title="NeuroLLM AI Disclaimer" />
    
    <div className="bg-purple-500/10 border border-purple-500/30 p-4 mb-6">
      <p className="text-purple-300 font-semibold text-sm">
        ⚠️ NeuroLLM is an experimental AI model trained by a decentralized network. 
        Outputs may be inaccurate, incomplete, biased, or harmful. Use at your own risk.
      </p>
    </div>

    <SubSection title="1. Experimental Nature">
      <p>
        NeuroLLM is trained from scratch by a decentralized network of participants. 
        Unlike commercial AI models:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>The model starts from random initialization ("untrained")</li>
        <li>Quality improves over time as the network trains</li>
        <li>Early outputs may be largely incoherent or "gibberish"</li>
        <li>There is no centralized quality control or curation</li>
        <li>Training data comes from network participants</li>
      </ul>
    </SubSection>

    <SubSection title="2. No Accuracy Guarantees">
      <p>
        NeuroLLM outputs may be:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li><strong>Factually incorrect</strong> - The model may generate false information</li>
        <li><strong>Outdated</strong> - Information may not reflect current reality</li>
        <li><strong>Incomplete</strong> - Responses may lack important context</li>
        <li><strong>Biased</strong> - Outputs may reflect biases in training data</li>
        <li><strong>Inconsistent</strong> - The model may contradict itself</li>
      </ul>
      <p className="mt-3">
        <strong className="text-white">Do not rely on NeuroLLM outputs</strong> for critical 
        decisions including medical, legal, financial, or safety-related matters.
      </p>
    </SubSection>

    <SubSection title="3. Harmful Content">
      <p>
        NeuroLLM may generate content that is:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Offensive, inappropriate, or disturbing</li>
        <li>Misleading or manipulative</li>
        <li>Potentially dangerous if acted upon</li>
        <li>Violating of third-party rights</li>
      </ul>
      <p className="mt-3">
        Due to the decentralized nature of the network, content filtering and safety 
        mechanisms may be limited or absent.
      </p>
    </SubSection>

    <SubSection title="4. User Responsibility">
      <p>
        By using NeuroLLM, you agree to:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Verify any important information from authoritative sources</li>
        <li>Not use outputs for illegal purposes</li>
        <li>Not rely on outputs for professional advice</li>
        <li>Take full responsibility for how you use the outputs</li>
        <li>Not hold NeuroShard liable for any harm caused by outputs</li>
      </ul>
    </SubSection>

    <SubSection title="5. Data Privacy">
      <p>
        Prompts and interactions with NeuroLLM may be:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Processed by multiple nodes in the network</li>
        <li>Used for training purposes (with privacy protections)</li>
        <li>Visible to network participants during processing</li>
      </ul>
      <p className="mt-3">
        <strong className="text-white">Do not submit sensitive personal information</strong>, 
        passwords, financial details, or confidential data to NeuroLLM.
      </p>
    </SubSection>

    <SubSection title="6. No Professional Advice">
      <p>
        NeuroLLM is <strong className="text-white">NOT</strong> a substitute for:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Medical professionals or healthcare providers</li>
        <li>Licensed attorneys or legal counsel</li>
        <li>Certified financial advisors</li>
        <li>Mental health professionals</li>
        <li>Any other qualified professional</li>
      </ul>
    </SubSection>
  </div>
);

const PrivacyPolicy = () => (
  <div>
    <SectionTitle icon={Lock} title="Privacy Policy" />
    
    <SubSection title="1. Information We Collect">
      <p><strong className="text-white">Account Information:</strong></p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Email address (for account creation and communication)</li>
        <li>Hashed password (we never store plain text passwords)</li>
        <li>Wallet public address (derived from your recovery phrase)</li>
      </ul>
      
      <p className="mt-3"><strong className="text-white">Usage Information:</strong></p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Node participation metrics (uptime, tokens processed)</li>
        <li>Transaction history on the NEURO ledger</li>
        <li>Proof of Neural Work contributions</li>
      </ul>
    </SubSection>

    <SubSection title="2. Information We Do NOT Collect">
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Your recovery phrase or private keys (stored only on your device)</li>
        <li>Raw training data (processed locally on nodes)</li>
        <li>Personal identification documents</li>
      </ul>
    </SubSection>

    <SubSection title="3. How We Use Information">
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Authenticate your account</li>
        <li>Process NEURO token transactions</li>
        <li>Verify Proof of Neural Work contributions</li>
        <li>Send important service notifications</li>
        <li>Improve the network and service</li>
      </ul>
    </SubSection>

    <SubSection title="4. Decentralized Network">
      <p>
        NeuroShard operates as a decentralized network. This means:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Ledger transactions are publicly visible on the network</li>
        <li>Your wallet address (not email) is associated with transactions</li>
        <li>Node operators may process data passing through their nodes</li>
        <li>No central entity has complete access to all network data</li>
      </ul>
    </SubSection>

    <SubSection title="5. Data Security">
      <p>
        We implement reasonable security measures including:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>HTTPS encryption for all web traffic</li>
        <li>Password hashing using industry-standard algorithms</li>
        <li>ECDSA cryptographic signatures for transactions</li>
        <li>JWT tokens for session authentication</li>
      </ul>
    </SubSection>

    <SubSection title="6. Your Rights">
      <p>You may:</p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Access your account information</li>
        <li>Update your email address</li>
        <li>Request deletion of your account</li>
      </ul>
      <p className="mt-3">
        Note: Ledger transactions cannot be deleted as they are part of the 
        decentralized record.
      </p>
    </SubSection>

    <SubSection title="7. Third-Party Services">
      <p>
        We may use third-party services for:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>Hosting and infrastructure</li>
        <li>Analytics (anonymized)</li>
        <li>Error monitoring</li>
      </ul>
    </SubSection>

    <SubSection title="8. Contact">
      <p>
        For privacy-related inquiries, please contact us through our GitHub repository.
      </p>
    </SubSection>
  </div>
);

const RiskDisclosure = () => (
  <div>
    <SectionTitle icon={AlertTriangle} title="Risk Disclosure" />
    
    <div className="bg-red-500/10 border border-red-500/30 p-4 mb-6">
      <p className="text-red-300 font-semibold text-sm">
        ⚠️ PARTICIPATING IN NEUROSHARD INVOLVES SIGNIFICANT RISKS. YOU COULD LOSE ALL 
        VALUE ASSOCIATED WITH YOUR PARTICIPATION. ONLY PARTICIPATE IF YOU FULLY 
        UNDERSTAND AND ACCEPT THESE RISKS.
      </p>
    </div>

    <SubSection title="1. Technology Risks">
      <ul className="list-disc list-inside space-y-2 ml-2">
        <li>
          <strong className="text-white">Software Bugs:</strong> The software may contain 
          bugs, vulnerabilities, or errors that could result in loss of tokens or data.
        </li>
        <li>
          <strong className="text-white">Security Vulnerabilities:</strong> Despite security 
          measures, the network could be vulnerable to hacks, exploits, or attacks.
        </li>
        <li>
          <strong className="text-white">Protocol Changes:</strong> The protocol may be 
          modified in ways that affect token value or functionality.
        </li>
        <li>
          <strong className="text-white">Smart Contract Risks:</strong> Bugs in the consensus 
          or reward mechanisms could result in incorrect token distributions.
        </li>
      </ul>
    </SubSection>

    <SubSection title="2. Network Risks">
      <ul className="list-disc list-inside space-y-2 ml-2">
        <li>
          <strong className="text-white">Byzantine Attacks:</strong> Up to 49% of nodes could 
          potentially be malicious, affecting network operation.
        </li>
        <li>
          <strong className="text-white">Model Poisoning:</strong> Malicious actors may attempt 
          to corrupt the AI model through poisoned gradients.
        </li>
        <li>
          <strong className="text-white">Network Failure:</strong> The network could fail, 
          become unusable, or be discontinued.
        </li>
        <li>
          <strong className="text-white">Centralization Risks:</strong> Despite decentralized 
          design, the network could become centralized.
        </li>
      </ul>
    </SubSection>

    <SubSection title="3. Economic Risks">
      <ul className="list-disc list-inside space-y-2 ml-2">
        <li>
          <strong className="text-white">Value Fluctuation:</strong> NEURO tokens may lose 
          all value. There is no guarantee of any value retention.
        </li>
        <li>
          <strong className="text-white">Liquidity Risk:</strong> You may not be able to 
          exchange NEURO tokens for other currencies or assets.
        </li>
        <li>
          <strong className="text-white">Staking Risk:</strong> Staked tokens may be slashed 
          for malicious behavior (even if unintentional).
        </li>
        <li>
          <strong className="text-white">Reward Volatility:</strong> Reward rates may change 
          based on network conditions and governance decisions.
        </li>
      </ul>
    </SubSection>

    <SubSection title="4. Regulatory Risks">
      <ul className="list-disc list-inside space-y-2 ml-2">
        <li>
          <strong className="text-white">Legal Status:</strong> The legal status of utility 
          tokens and decentralized AI networks is uncertain and evolving.
        </li>
        <li>
          <strong className="text-white">Regulatory Action:</strong> Government action could 
          restrict, prohibit, or require licensing for network participation.
        </li>
        <li>
          <strong className="text-white">Tax Treatment:</strong> Tax treatment of token 
          earnings is unclear and may result in unexpected tax liabilities.
        </li>
      </ul>
    </SubSection>

    <SubSection title="5. Operational Risks">
      <ul className="list-disc list-inside space-y-2 ml-2">
        <li>
          <strong className="text-white">Wallet Loss:</strong> Losing your recovery phrase 
          means permanent loss of access to your tokens.
        </li>
        <li>
          <strong className="text-white">Hardware Requirements:</strong> Running a node 
          requires computing resources and may incur electricity costs.
        </li>
        <li>
          <strong className="text-white">Downtime:</strong> Network or node downtime could 
          affect your ability to earn or use tokens.
        </li>
      </ul>
    </SubSection>

    <SubSection title="6. AI Model Risks">
      <ul className="list-disc list-inside space-y-2 ml-2">
        <li>
          <strong className="text-white">Output Quality:</strong> The AI model may produce 
          low-quality, incorrect, or harmful outputs.
        </li>
        <li>
          <strong className="text-white">Training Quality:</strong> Decentralized training 
          may not achieve the quality of centralized alternatives.
        </li>
        <li>
          <strong className="text-white">Misuse:</strong> The AI model could be misused for 
          harmful purposes.
        </li>
      </ul>
    </SubSection>

    <SubSection title="7. Acknowledgment">
      <p>
        By using NeuroShard, you acknowledge that:
      </p>
      <ul className="list-disc list-inside space-y-1 ml-2">
        <li>You have read and understood these risks</li>
        <li>You are participating voluntarily</li>
        <li>You can afford to lose any value you contribute</li>
        <li>You will not hold NeuroShard or its contributors liable for losses</li>
        <li>You are responsible for your own due diligence</li>
      </ul>
    </SubSection>
  </div>
);

