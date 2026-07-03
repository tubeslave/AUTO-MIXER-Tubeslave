import React, { useState, useEffect, useCallback } from 'react';
import websocketService from '../services/websocket';
import './ConnectionBanner.css';

/**
 * Глобальный индикатор состояния для live-работы:
 * - полноэкранный баннер при потере бэкенда (heartbeat) или пульта;
 * - кнопка EMERGENCY STOP (останавливает автоматику записи на пульт);
 * - уведомление о hot-reload конфига.
 *
 * Самодостаточен: сам подписывается на события websocketService
 * и снимает подписки при размонтировании.
 */
function ConnectionBanner() {
  const [backendState, setBackendState] = useState('connecting');
  const [mixerConnected, setMixerConnected] = useState(false);
  const [stopPending, setStopPending] = useState(false);
  const [stopApplied, setStopApplied] = useState(false);
  const [toast, setToast] = useState(null);

  useEffect(() => {
    const onConnectionState = (data) => {
      setBackendState(data.state);
      setMixerConnected(Boolean(data.mixerConnected));
    };
    const onConnected = () => setBackendState('connected');
    const onDisconnected = () => setBackendState('reconnecting');
    const onEmergencyApplied = () => {
      setStopPending(false);
      setStopApplied(true);
    };
    const onConfigReloaded = (data) => {
      setToast(`Конфигурация перезагружена (${data.source || 'automixer.yaml'})`);
      setTimeout(() => setToast(null), 5000);
    };
    const onConnectionStatus = (data) => {
      if (typeof data.connected === 'boolean') setMixerConnected(data.connected);
    };

    websocketService.on('connection_state', onConnectionState);
    websocketService.on('connected', onConnected);
    websocketService.on('disconnected', onDisconnected);
    websocketService.on('emergency_stop_applied', onEmergencyApplied);
    websocketService.on('config_reloaded', onConfigReloaded);
    websocketService.on('connection_status', onConnectionStatus);
    return () => {
      websocketService.off('connection_state', onConnectionState);
      websocketService.off('connected', onConnected);
      websocketService.off('disconnected', onDisconnected);
      websocketService.off('emergency_stop_applied', onEmergencyApplied);
      websocketService.off('config_reloaded', onConfigReloaded);
      websocketService.off('connection_status', onConnectionStatus);
    };
  }, []);

  const handleEmergencyStop = useCallback(() => {
    setStopPending(true);
    setStopApplied(false);
    websocketService.send({ type: 'emergency_stop' });
    // Если бэкенд не подтвердил за 3 с — показываем, что стоп НЕ применён
    setTimeout(() => {
      setStopPending((pending) => {
        if (pending) setToast('⚠ EMERGENCY STOP не подтверждён бэкендом!');
        return false;
      });
    }, 3000);
  }, []);

  const backendDown = backendState === 'reconnecting' || backendState === 'unauthorized';

  return (
    <>
      {backendDown && (
        <div className="conn-banner conn-banner-critical" role="alert">
          {backendState === 'unauthorized'
            ? '⛔ ДОСТУП ОТКЛОНЁН: неверный токен (automixer_token)'
            : '🔌 СВЯЗЬ С БЭКЕНДОМ ПОТЕРЯНА — переподключение… Автоматика недоступна, пульт продолжает работать в ручном режиме'}
        </div>
      )}
      {!backendDown && backendState === 'connected' && !mixerConnected && (
        <div className="conn-banner conn-banner-warning" role="alert">
          ⚠ Пульт не подключён — команды не достигают железа
        </div>
      )}
      {stopApplied && (
        <div className="conn-banner conn-banner-stopped" role="status">
          🛑 EMERGENCY STOP применён: автоматическая запись на пульт остановлена
          <button className="conn-banner-dismiss" onClick={() => setStopApplied(false)}>×</button>
        </div>
      )}
      {toast && <div className="conn-banner conn-banner-info" role="status">{toast}</div>}
      <button
        className={`emergency-stop-btn ${stopPending ? 'pending' : ''}`}
        onClick={handleEmergencyStop}
        disabled={backendDown}
        title="Немедленно остановить автоматическую запись на пульт"
      >
        {stopPending ? '…' : 'STOP'}
      </button>
    </>
  );
}

export default ConnectionBanner;
