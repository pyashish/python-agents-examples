export interface AppConfig {
  pageTitle: string;
  pageDescription: string;
  companyName: string;

  supportsChatInput: boolean;
  supportsVideoInput: boolean;
  supportsScreenShare: boolean;
  isPreConnectBufferEnabled: boolean;

  logo: string;
  startButtonText: string;
  accent?: string;
  logoDark?: string;
  accentDark?: string;

  agentName?: string;
  sandboxId?: string;
}

export const APP_CONFIG_DEFAULTS: AppConfig = {
  companyName: 'Deepavali',
  pageTitle: 'Kannada Deepavali Voice',
  pageDescription: 'Speak in Kannada with a LemonSlice avatar — ದೀಪಾವಳಿ ಶುಭಾಶಯಗಳು',

  supportsChatInput: true,
  supportsVideoInput: false,
  supportsScreenShare: false,
  isPreConnectBufferEnabled: true,

  logo: '/lk-logo.svg',
  accent: '#f59e0b',
  logoDark: '/lk-logo-dark.svg',
  accentDark: '#fbbf24',
  startButtonText: 'Call our Surprise Secret Special friend',

  agentName: process.env.AGENT_NAME ?? 'kannada-deepavali',
  sandboxId: undefined,
};