import test from 'node:test';
import assert from 'node:assert/strict';

import {
  assertWebSocketConnectionPolicy,
  deriveSecureRemoteWebSocketUrl,
  isLoopbackHostname,
  remoteAccessPolicy
} from '../remoteAuthPolicy.mjs';

test('remote access policy distinguishes exact loopback, insecure LAN, and HTTPS remote', () => {
  assert.equal(isLoopbackHostname('localhost'), true);
  assert.equal(isLoopbackHostname('127.0.0.1'), true);
  assert.equal(isLoopbackHostname('::1'), true);
  assert.equal(isLoopbackHostname('localhost.example'), false);

  assert.equal(remoteAccessPolicy({
    protocol: 'http:',
    hostname: '127.0.0.1',
    host: '127.0.0.1:3000'
  }).kind, 'local');
  assert.equal(remoteAccessPolicy({
    protocol: 'http:',
    hostname: '192.168.1.20',
    host: '192.168.1.20:3000'
  }).kind, 'blocked');
  const secure = remoteAccessPolicy({
    protocol: 'https:',
    hostname: 'mix.example',
    host: 'mix.example'
  });
  assert.equal(secure.kind, 'remote');
  assert.equal(secure.websocketUrl, 'wss://mix.example/ws');
});

test('secure remote endpoint is same-origin WSS without credentials or URL data', () => {
  const locationLike = {
    protocol: 'https:',
    hostname: 'mix.example',
    host: 'mix.example'
  };
  assert.equal(deriveSecureRemoteWebSocketUrl(locationLike), 'wss://mix.example/ws');
  assert.doesNotThrow(() => assertWebSocketConnectionPolicy({
    url: 'wss://mix.example/ws',
    authToken: 'r'.repeat(43),
    locationLike
  }));
  for (const url of [
    'ws://mix.example/ws',
    'wss://other.example/ws',
    'wss://mix.example/other',
    'wss://user:pass@mix.example/ws',
    'wss://mix.example/ws?token=leak',
    'wss://mix.example/ws#fragment'
  ]) {
    assert.throws(() => assertWebSocketConnectionPolicy({
      url,
      authToken: 'r'.repeat(43),
      locationLike
    }));
  }
  assert.throws(() => assertWebSocketConnectionPolicy({
    url: 'wss://mix.example/ws',
    authToken: '',
    locationLike
  }));
  assert.throws(() => assertWebSocketConnectionPolicy({
    url: 'ws://127.0.0.1:8765',
    authToken: 'r'.repeat(43),
    locationLike
  }));
});
