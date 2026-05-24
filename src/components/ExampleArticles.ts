export interface ExampleArticle {
  title: string;
  sourceUrl?: string;
  text: string;
  badge: "Real News" | "Clickbait / Fake" | "Mixed Rumor";
}

export const EXAMPLE_ARTICLES: ExampleArticle[] = [
  {
    title: "NASA James Webb Space Telescope Detects Water Vapor in Rocky Planet Zone",
    sourceUrl: "https://www.nasa.gov/news",
    text: "Astronomers using NASA’s James Webb Space Telescope have made a breakthrough observation, detecting water vapor in the inner rocky region of a protoplanetary disk around a young star. The star, known as PDS 70, is located approximately 370 light-years from Earth in the constellation Centaurus. This region is typically host to rocky, terrestrial planets of similar mass to Earth. According to astrophysicists, the presence of water vapor suggests that rocky worlds forming there could have access to water reservoirs early in their development, substantially increasing the long-term potential for habitability. Standard spectrometer instruments on the James Webb Telescope confirmed the spectral signatures of water molecules with high confidence.",
    badge: "Real News"
  },
  {
    title: "SHOCKING TRUTH: Miracle Mountain Herb Cures Diabetes in 48 Hours, FDA Hides Medical Breakthrough!",
    sourceUrl: "https://www.healthsecretsunleashed-insiders.org",
    text: "BREAKING NEWS! An anonymous whistleblower has just exposed a miracle mountain herb that can completely reverse diabetes in less than 48 hours! Secret laboratory testing has proved that the ancient Himalayan flower 'Glyco-Cure' instantly repairs pancreatic cells. Yet, elite medical executives and corrupt FDA authorities are actively hiding this information from the general public. Shockingly, the mainstream media is completely silent on this! Millions are suffering while pharmaceutical companies rake in billions in drug profits. This is a massive cover-up! Local doctors are terrified of speaking out, but our source has confirmed that a hidden underground clinic is distributing the plant directly to insiders close to the project. Forward this to everyone before this page is deleted by big tech sensors!",
    badge: "Clickbait / Fake"
  },
  {
    title: "Rumor: Social Media Platform to Charge $10 Monthly Fee Starting Next Sunday, Netizens Skeptical",
    sourceUrl: "https://www.viral-tech-rumors.net",
    text: "An unverified rumor circulating across multiple internet forums suggests that a major social media platform might start charging a subscription fee of ten dollars per month as early as next Sunday. A leaked email screenshot posted by a user claiming to be an insider sparked widespread alarm, showing a draft announcement stating 'Premium Access Core Launch'. However, the company has released no official press statement, and their official PR spokesperson has not responded to inquiries. Digital privacy watchdogs argue that a sudden platform-wide paywall is highly unlikely and would lead to an immediate user migration. While some analysts believe a partial monetization strategy might be under review due to falling advertising revenues, the specific Sunday timeline lacks definitive supporting evidence. Netizens remain skeptical, with many calling out the leaked image as a potential deepfake or promotional hoax.",
    badge: "Mixed Rumor"
  }
];
