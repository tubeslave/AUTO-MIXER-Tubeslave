const LOOPBACK_HOSTNAMES = new Set(['localhost', '127.0.0.1', '::1', '[::1]']);

function normalizedHostname(value) {
  return String(value || '').trim().toLowerCase();
}

export function isLoopbackHostname(hostname) {
  return LOOPBACK_HOSTNAMES.has(normalizedHostname(hostname));
}

export function deriveSecureRemoteWebSocketUrl(locationLike = globalThis.location) {
  if (!locationLike || locationLike.protocol !== 'https:' || !locationLike.host) {
    throw new Error('Secure remote access requires an HTTPS page');
  }
  if (isLoopbackHostname(locationLike.hostname)) {
    throw new Error('Loopback pages do not use remote enrollment');
  }
  const endpoint = new URL(`wss://${locationLike.host}/ws`);
  if (
    endpoint.hostname !== normalizedHostname(locationLike.hostname)
    || endpoint.username
    || endpoint.password
    || endpoint.search
    || endpoint.hash
  ) {
    throw new Error('Secure remote WebSocket endpoint is invalid');
  }
  return endpoint.toString();
}

export function remoteAccessPolicy(locationLike = globalThis.location) {
  if (!locationLike?.hostname) {
    return {
      kind: 'local',
      requiresAuthentication: false,
      websocketUrl: 'ws://localhost:8765'
    };
  }
  if (isLoopbackHostname(locationLike.hostname)) {
    return {
      kind: 'local',
      requiresAuthentication: false,
      websocketUrl: locationLike.protocol === 'https:'
        ? `wss://${locationLike.host}/ws`
        : `ws://${locationLike.hostname}:8765`
    };
  }
  if (locationLike.protocol !== 'https:') {
    return {
      kind: 'blocked',
      requiresAuthentication: true,
      reason: 'Remote access is blocked on an insecure page. Open the HTTPS operator URL.'
    };
  }
  return {
    kind: 'remote',
    requiresAuthentication: true,
    websocketUrl: deriveSecureRemoteWebSocketUrl(locationLike)
  };
}

export function assertWebSocketConnectionPolicy({
  url,
  authToken = '',
  locationLike = globalThis.location
}) {
  let endpoint;
  try {
    endpoint = new URL(String(url || ''));
  } catch {
    throw new Error('WebSocket connection policy blocked an invalid endpoint');
  }
  if (
    !['ws:', 'wss:'].includes(endpoint.protocol)
    || endpoint.username
    || endpoint.password
    || endpoint.search
    || endpoint.hash
  ) {
    throw new Error('WebSocket connection policy blocked an invalid endpoint');
  }
  if (isLoopbackHostname(endpoint.hostname)) {
    if (!locationLike?.hostname || isLoopbackHostname(locationLike.hostname)) return true;
    throw new Error('WebSocket connection policy blocked a remote-to-loopback endpoint');
  }

  if (
    !locationLike
    || locationLike.protocol !== 'https:'
    || isLoopbackHostname(locationLike.hostname)
    || endpoint.protocol !== 'wss:'
    || endpoint.host !== locationLike.host
    || endpoint.pathname !== '/ws'
    || !/^[A-Za-z0-9_-]{43}$/.test(String(authToken || ''))
  ) {
    throw new Error('WebSocket connection policy blocked insecure remote access');
  }
  return true;
}
