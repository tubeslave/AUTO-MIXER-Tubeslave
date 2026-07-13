import { FormEvent, ReactNode, useEffect, useMemo, useState } from 'react';
import websocketService from '../services/websocket';
import { remoteAccessPolicy } from '../services/remoteAuthPolicy.mjs';

type RemoteAuthGateProps = {
  children: ReactNode;
};

export default function RemoteAuthGate({ children }: RemoteAuthGateProps) {
  const policy = useMemo(() => remoteAccessPolicy(window.location), []);
  const [authorized, setAuthorized] = useState(policy.kind === 'local');
  const [token, setToken] = useState('');
  const [pending, setPending] = useState(false);
  const [status, setStatus] = useState('Enter the operator token for this session.');

  useEffect(() => {
    if (policy.kind !== 'remote') return undefined;
    const handleAuthenticationFailure = () => {
      websocketService.setAuthToken('', { reconnect: false });
      setAuthorized(false);
      setPending(false);
      setStatus('Token rejected or expired. Enter the current operator token.');
    };
    websocketService.on('authentication_failed', handleAuthenticationFailure);
    return () => websocketService.off('authentication_failed', handleAuthenticationFailure);
  }, [policy.kind]);

  if (policy.kind === 'local' || authorized) return children;

  if (policy.kind === 'blocked') {
    return (
      <main className="grid min-h-screen place-items-center bg-bg p-6 text-text">
        <section className="w-full max-w-md rounded-[8px] border border-danger/40 bg-panel p-5 shadow-glow">
          <div className="am-label text-danger">Remote access blocked</div>
          <h1 className="mt-2 text-lg font-bold">HTTPS is required</h1>
          <p className="mt-3 text-sm leading-6 text-muted">{policy.reason}</p>
          <p className="mt-3 text-xs leading-5 text-dim">
            No token field is shown on an insecure page, and no WebSocket connection is attempted.
          </p>
        </section>
      </main>
    );
  }

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (pending) return;
    const candidate = token.trim();
    setToken('');
    if (!/^[A-Za-z0-9_-]{43}$/.test(candidate)) {
      setStatus('The operator token is incomplete. Request the current token from the Mac operator.');
      return;
    }

    setPending(true);
    setStatus('Authenticating secure remote session…');
    websocketService.disconnect();
    websocketService.setUrl(policy.websocketUrl, {
      reconnect: false
    });
    websocketService.setAuthToken(candidate, {
      reconnect: false
    });
    try {
      await websocketService.connect();
      setAuthorized(true);
      setStatus('Authenticated.');
    } catch {
      websocketService.setAuthToken('', { reconnect: false });
      setAuthorized(false);
      setStatus('Token rejected or expired. Enter the current operator token.');
    } finally {
      setPending(false);
    }
  };

  return (
    <main className="grid min-h-screen place-items-center bg-bg p-6 text-text">
      <form
        className="w-full max-w-md rounded-[8px] border border-audio/35 bg-panel p-5 shadow-glow"
        onSubmit={submit}
      >
        <div className="am-label text-audio">Secure remote enrollment</div>
        <h1 className="mt-2 text-lg font-bold">AUTO MIXER operator access</h1>
        <p className="mt-2 text-xs leading-5 text-muted">
          The WSS endpoint is derived from this HTTPS page. The token is kept only in memory
          and is removed on reload or authentication failure.
        </p>
        <label className="mt-4 block text-xs font-semibold text-muted" htmlFor="operator-token">
          Operator token
        </label>
        <input
          id="operator-token"
          type="password"
          autoComplete="off"
          spellCheck={false}
          value={token}
          onChange={event => setToken(event.target.value)}
          disabled={pending}
          className="mt-2 w-full rounded-[5px] border border-line bg-bg px-3 py-2 font-mono text-sm text-text outline-none focus:border-audio disabled:opacity-60"
        />
        <p aria-live="polite" className="mt-3 min-h-5 text-xs text-muted">{status}</p>
        <button
          type="submit"
          disabled={pending}
          className="am-button mt-3 w-full border-audio/40 bg-audio/10 text-audio disabled:cursor-not-allowed disabled:opacity-60"
        >
          {pending ? 'Authenticating…' : 'Start secure session'}
        </button>
      </form>
    </main>
  );
}
