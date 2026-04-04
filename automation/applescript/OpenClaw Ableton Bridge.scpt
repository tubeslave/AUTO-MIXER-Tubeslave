-- OpenClaw Ableton Bridge
-- Интеграция OpenClaw с Ableton Live для AUTO-MIXER Tubeslave

property openclawPath : "/Users/" & (do shell script "whoami") & "/.nvm/versions/node/v24.11.1/bin/openclaw"
property automixerPath : "/Users/" & (do shell script "whoami") & "/AUTO-MIXER-Tubeslave-main"

-- Запуск AUTO-MIXER через start.py
on startAutomixer()
	try
		tell application "Terminal"
			set newTab to do script "cd " & quoted form of automixerPath & " && python3 start.py"
			activate
		end tell
		display notification "AUTO-MIXER запущен" with title "OpenClaw Bridge"
		return true
	on error errMsg
		display dialog "Ошибка запуска AUTO-MIXER: " & errMsg buttons {"OK"} with icon stop
		return false
	end try
end startAutomixer

-- Отправка OSC команды через OpenClaw
on sendOscCommand(command)
	try
		set fullCmd to "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && cd " & quoted form of automixerPath & "/backend && python3 -c \"import asyncio; from wing_client import WingClient; c = WingClient(); asyncio.run(c.connect()); asyncio.run(c.send_command('" & command & "'))\""
		
		do shell script fullCmd
		display notification "OSC команда отправлена: " & command with title "OpenClaw Bridge"
	return true
	on error errMsg
		display dialog "Ошибка OSC: " & errMsg buttons {"OK"} with icon stop
		return false
	end try
end sendOscCommand

-- Открытие Ableton Live проекта
on openAbletonProject(projectPath)
	try
		tell application "Ableton Live"
			if not running then launch
			delay 2
			open projectPath
			activate
		end tell
		display notification "Проект открыт: " & projectPath with title "OpenClaw Bridge"
	return true
	on error
		-- Пробуем через Finder
		try
			tell application "Finder"
				open file projectPath
			end tell
			return true
		on error errMsg
			display dialog "Ошибка открытия проекта: " & errMsg buttons {"OK"} with icon stop
			return false
		end try
	end try
end openAbletonProject

-- Настройка Audio MIDI Setup
on configureAudioMidi()
	try
		tell application "Audio MIDI Setup"
			activate
			delay 1
			-- Открыть Audio Devices
			tell application "System Events"
				keystroke "2" using {command down, option down}
			end tell
		end tell
		display notification "Audio MIDI Setup открыт" with title "OpenClaw Bridge"
	return true
	on error errMsg
		display dialog "Ошибка: " & errMsg buttons {"OK"} with icon stop
		return false
	end try
end configureAudioMidi

-- Полный сетап для live sound
on fullLiveSetup()
	-- Шаг 1: Audio MIDI Setup
	configureAudioMidi()
	delay 2
	
	-- Шаг 2: Запуск AUTO-MIXER
	startAutomixer()
	delay 3
	
	-- Шаг 3: Проверка OpenClaw Gateway
	try
		do shell script "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && " & openclawPath & " health"
		display notification "OpenClaw Gateway активен" with title "Setup"
	on error
		display dialog "OpenClaw Gateway не запущен. Запустить?" buttons {"Да", "Нет"} default button "Да"
		if button returned of result is "Да" then
			do shell script "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && nohup " & openclawPath & " gateway > /dev/null 2>&1 &"
			delay 3
		end if
	end try
	
	display notification "Live Sound Setup завершен!" with title "OpenClaw Bridge"
end fullLiveSetup

-- Основной интерфейс
on run
	set options to {"🎛 Полный сетап (Audio MIDI + AUTO-MIXER + OpenClaw)", "🚀 Запустить AUTO-MIXER", "🎵 Открыть Ableton Live проект", "🔊 Audio MIDI Setup", "📡 Отправить OSC команду", "🔍 Проверить статус", "🤖 OpenClaw команды"}
	
	set choice to choose from list options with title "OpenClaw Ableton Bridge" with prompt "Выберите действие:" OK button name "Выполнить" cancel button name "Отмена"
	
	if choice is false then return
	
	set selected to item 1 of choice
	
	if selected contains "Полный сетап" then
		fullLiveSetup()
		
	else if selected contains "Запустить AUTO-MIXER" then
		startAutomixer()
		
	else if selected contains "Открыть Ableton Live проект" then
		set projectPath to choose file with prompt "Выберите Ableton проект (.als):" of type {"com.ableton.live.project"}
		openAbletonProject(POSIX path of projectPath)
		
	else if selected contains "Audio MIDI Setup" then
		configureAudioMidi()
		
	else if selected contains "Отправить OSC команду" then
		set oscCmd to text returned of (display dialog "OSC команда:" default answer "/ch/01/mix/fader 0.75")
		sendOscCommand(oscCmd)
		
	else if selected contains "Проверить статус" then
		checkAllStatus()
		
	else if selected contains "OpenClaw команды" then
		showOpenClawCommands()
	end if
end run

-- Проверка всех компонентов
on checkAllStatus()
	set statusText to "=== СТАТУС СИСТЕМЫ ===" & return & return
	
	-- OpenClaw
	try
		do shell script "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && " & openclawPath & " health"
		set statusText to statusText & "✅ OpenClaw Gateway: ОК" & return
	on error
		set statusText to statusText & "❌ OpenClaw Gateway: Не запущен" & return
	end try
	
	-- AUTO-MIXER Backend
	try
		do shell script "lsof -i :8765 | grep LISTEN"
		set statusText to statusText & "✅ AUTO-MIXER Backend: ОК (port 8765)" & return
	on error
		set statusText to statusText & "❌ AUTO-MIXER Backend: Не запущен" & return
	end try
	
	-- Ableton Live
	try
		tell application "System Events"
			if exists (process "Live") then
				set statusText to statusText & "✅ Ableton Live: Запущен" & return
			else
				set statusText to statusText & "❌ Ableton Live: Не запущен" & return
			end if
		end tell
	on error
		set statusText to statusText & "❌ Ableton Live: Неизвестно" & return
	end try
	
	-- WING Connection (если настроено)
	try
		set pingResult to do shell script "ping -c 1 -W 1 192.168.1.1 2>/dev/null && echo 'OK' || echo 'FAIL'"
		if pingResult contains "OK" then
			set statusText to statusText & "✅ WING Mixer: Доступен (192.168.1.1)" & return
		else
			set statusText to statusText & "⚠️ WING Mixer: Не доступен" & return
		end if
	on error
		set statusText to statusText & "⚠️ WING Mixer: Проверка недоступна" & return
	end try
	
	display dialog statusText buttons {"OK", "Обновить"} default button "OK"
	if button returned of result is "Обновить" then
		checkAllStatus()
	end if
end checkAllStatus

-- Показать OpenClaw команды
on showOpenClawCommands()
	set commands to {"gateway status", "health", "dashboard", "tui", "agent --help", "message send", "logs"}
	set chosen to choose from list commands with title "OpenClaw команды" OK button name "Выполнить" cancel button name "Назад"
	
	if chosen is false then
		run
		return
	end if
	
	set cmd to item 1 of chosen
	
	if cmd is "dashboard" or cmd is "tui" then
		tell application "Terminal"
			do script "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && openclaw " & cmd
			activate
		end tell
	else
		try
			set result to do shell script "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && openclaw " & cmd
			display dialog "Результат:" & return & return & result buttons {"OK", "Ещё команды"} default button "OK"
			if button returned of result is "Ещё команды" then
				showOpenClawCommands()
			end if
		on error errMsg
			display dialog "Ошибка: " & errMsg buttons {"OK"} with icon stop
		end try
	end if
end showOpenClawCommands
