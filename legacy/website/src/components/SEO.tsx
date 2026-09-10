import { Helmet } from 'react-helmet-async';

interface SEOProps {
  title?: string;
  description?: string;
  name?: string;
  type?: string;
  image?: string;
  url?: string;
}

export const SEO = ({
  title,
  description,
  name = "NeuroShard",
  type = "website",
  image = "https://neuroshard.ai/logo_large.png",
  url = "https://neuroshard.ai"
}: SEOProps) => {
  const siteTitle = "NeuroShard - The Decentralized 'Living' AI";
  const defaultDescription = "Join the NeuroShard network. A decentralized, dynamic AI ecosystem where computation is currency.";

  const finalTitle = title ? `${title} | ${name}` : siteTitle;
  const finalDescription = description || defaultDescription;

  return (
    <Helmet>
      {/* Standard metadata tags */}
      <title>{finalTitle}</title>
      <meta name="description" content={finalDescription} />
      <link rel="canonical" href={url} />

      {/* Facebook tags */}
      <meta property="og:type" content={type} />
      <meta property="og:title" content={finalTitle} />
      <meta property="og:description" content={finalDescription} />
      <meta property="og:image" content={image} />
      <meta property="og:url" content={url} />

      {/* Twitter tags */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:creator" content={name} />
      <meta name="twitter:title" content={finalTitle} />
      <meta name="twitter:description" content={finalDescription} />
      <meta name="twitter:image" content={image} />
    </Helmet>
  );
};
