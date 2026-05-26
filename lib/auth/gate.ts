const ALWAYS_ALLOW_EXACT = new Set(["/login", "/api/login", "/api/logout", "/favicon.ico", "/robots.txt"]);

export type GateInput = {
  pathname: string;
  hasValidSession: boolean;
  appPassword: string | undefined;
  sessionSecret: string | undefined;
  onVercel: boolean;
};

export type GateDecision = { allow: true } | { allow: false; reason: "login" | "misconfig" };

export function shouldAllow(input: GateInput): GateDecision {
  const { pathname, hasValidSession, appPassword, sessionSecret, onVercel } = input;

  if (ALWAYS_ALLOW_EXACT.has(pathname) || pathname.startsWith("/_next/") || pathname.startsWith("/static/")) {
    return { allow: true };
  }

  const configured = Boolean(appPassword) && Boolean(sessionSecret);
  if (!configured) {
    return onVercel ? { allow: false, reason: "misconfig" } : { allow: true };
  }

  return hasValidSession ? { allow: true } : { allow: false, reason: "login" };
}
