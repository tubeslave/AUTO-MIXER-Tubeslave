-- OpenClaw Launcher
-- Запускает OpenClaw Gateway и предоставляет доступ к управлению приложениями
-- Местоположение: ~/Library/Scripts/ или Script Menu

property openclawPath : "/Users/" & (do shell script "whoami") & "/.nvm/versions/node/v24.11.1/bin/openclaw"
property gatewayPort : "18789"
property gatewayToken : "7a3d22f4ccfca79a966e63251569ccf1796e562634eab8c7"

-- Проверка и запуск Gateway
on startGateway()
	try
		do shell script "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && " & openclawPath & " gateway status"
		display notification "OpenClaw Gateway уже запущен" with title "OpenClaw"
	return true
	on error
		display notification "Запуск OpenClaw Gateway..." with title "OpenClaw"
		do shell script "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && nohup " & openclawPath & " gateway > /dev/null 2>&1 &"
		delay 3
	return true
	end try
end startGateway

-- Отправка команды через OpenClaw
on sendCommand(commandText)
	try
		set cmd to "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && " & openclawPath & " agent --message " & quoted form of commandText
		set result to do shell script cmd
		display dialog "Ответ агента:" & return & return & result buttons {"OK"} default button "OK"
	on error errMsg
		display dialog "Ошибка: " & errMsg buttons {"OK"} default button "OK" with icon stop
	end try
end sendCommand

-- Управление приложениями через AppleScript
on controlApplication(appName, action)
	try
		tell application appName
			if action is "activate" then
				activate
			else if action is "quit" then
				quit
			else if action is "hide" then
				run script "tell application \"System Events\" to set visible of process \"" & appName & "\" to false"
			else if action is "show" then
				run script "tell application \"System Events\" to set visible of process \"" & appName & "\" to true"
			end if
		end tell
		display notification "Выполнено: " & action & " " & appName with title "OpenClaw"
	on error errMsg
		display notification "Ошибка: " & errMsg with title "OpenClaw"
	end try
end controlApplication

-- Основной диалог
on run
	-- Запускаем Gateway если нужно
	if not startGateway() then return
	
	-- Меню выбора действия
	set actionList to {"Отправить команду агенту", "Открыть приложение", "Закрыть приложение", "Скрыть приложение", "Показать приложение", "Открыть терминал", "Проверить статус"}
	set chosenAction to choose from list actionList with title "OpenClaw Automation" with prompt "Выберите действие:" default items {"Отправить команду агенту"} OK button name "Выполнить" cancel button name "Отмена"
	
	if chosenAction is false then return
	
	set selectedAction to item 1 of chosenAction
	
	if selectedAction is "Отправить команду агенту" then
		set userInput to display dialog "Введите команду для OpenClaw:" default answer "" buttons {"Отмена", "Отправить"} default button "Отправить"
		if button returned of userInput is "Отправить" then
			sendCommand(text returned of userInput)
		end if
		
	else if selectedAction is "Открыть приложение" then
		set appList to {"Terminal", "Safari", "Finder", "Music", "Ableton Live", "Logic Pro", "Audio MIDI Setup", "Console"}
		set chosenApp to choose from list appList with title "Выберите приложение" OK button name "Открыть" cancel button name "Отмена"
		if chosenApp is not false then
			controlApplication(item 1 of chosenApp, "activate")
		end if
		
	else if selectedAction is "Закрыть приложение" then
		set appList to {"Terminal", "Safari", "Music", "Console"}
		set chosenApp to choose from list appList with title "Выберите приложение для закрытия" OK button name "Закрыть" cancel button name "Отмена"
		if chosenApp is not false then
			controlApplication(item 1 of chosenApp, "quit")
		end if
		
	else if selectedAction is "Открыть терминал" then
		tell application "Terminal"
			do script "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && openclaw --help"
			activate
		end tell
		
	else if selectedAction is "Проверить статус" then
		try
			set status to do shell script "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && " & openclawPath & " health"
			display dialog "Статус OpenClaw:" & return & return & status buttons {"OK"} default button "OK"
		on error errMsg
			display dialog "Gateway не запущен или ошибка: " & errMsg buttons {"OK"} default button "OK" with icon stop
		end try
	end if
end run
