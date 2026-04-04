-- OpenClaw Application Bridge
-- Позволяет OpenClaw управлять macOS приложениями через AppleScript

-- Получение списка запущенных приложений
on getRunningApps()
	tell application "System Events"
		set appList to name of every application process whose background only is false
	end tell
	return appList
end getRunningApps

-- Активация приложения
on activateApp(appName)
	try
		tell application appName to activate
		return true
	on error
		return false
	end try
end activateApp

-- Закрытие приложения
on quitApp(appName)
	try
		tell application appName to quit
		return true
	on error
		return false
	end try
end quitApp

-- Выполнение AppleScript кода через shell
on runAppleScript(scriptText)
	try
		set result to run script scriptText
		return result as string
	on error errMsg
		return "Ошибка: " & errMsg
	end try
end runAppleScript

-- Интеграция с OpenClaw - отправка результата
on notifyOpenClaw(message)
	try
		do shell script "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && openclaw message send --message " & quoted form of message
	on error
		-- Если не удалось отправить, просто показываем уведомление
		display notification message with title "OpenClaw Bridge"
	end try
end notifyOpenClaw

-- Основной интерфейс
on run
	set options to {"Показать запущенные приложения", "Активировать приложение", "Закрыть приложение", "Выполнить AppleScript", "Создать быструю команду"}
	set choice to choose from list options with title "OpenClaw Application Bridge" OK button name "Выбрать" cancel button name "Отмена"
	
	if choice is false then return
	
	set selected to item 1 of choice
	
	if selected is "Показать запущенные приложения" then
		set apps to getRunningApps()
		set appText to ""
		repeat with appName in apps
			set appText to appText & appName & return
		end repeat
		display dialog "Запущенные приложения:" & return & return & appText buttons {"OK", "Копировать"} default button "OK"
		if button returned of result is "Копировать" then
			set the clipboard to appText
		end if
		
	else if selected is "Активировать приложение" then
		set appName to text returned of (display dialog "Введите имя приложения:" default answer "Terminal")
		if activateApp(appName) then
			display notification "Приложение " & appName & " активировано" with title "OpenClaw Bridge"
		else
			display dialog "Не удалось активировать " & appName with icon stop
		end if
		
	else if selected is "Закрыть приложение" then
		set appName to text returned of (display dialog "Введите имя приложения для закрытия:" default answer "")
		if quitApp(appName) then
			display notification "Приложение " & appName & " закрыто" with title "OpenClaw Bridge"
		else
			display dialog "Не удалось закрыть " & appName with icon stop
		end if
		
	else if selected is "Выполнить AppleScript" then
		set scriptText to text returned of (display dialog "Введите AppleScript код:" default answer "tell application \"Finder\" to activate")
		set result to runAppleScript(scriptText)
		display dialog "Результат:" & return & return & result buttons {"OK"} default button "OK"
		
	else if selected is "Создать быструю команду" then
		createQuickCommand()
	end if
end run

-- Создание быстрой команды на рабочем столе
on createQuickCommand()
	set cmdName to text returned of (display dialog "Имя команды:" default answer "Launch Terminal")
	set cmdAction to text returned of (display dialog "AppleScript действие:" default answer "tell application \"Terminal\" to activate")
	
	set desktopPath to (path to desktop) as string
	set filePath to desktopPath & cmdName & ".scpt"
	
	try
		tell application "Script Editor"
			set newScript to make new document
			set text of newScript to cmdAction
			save newScript in filePath as "script"
			close newScript
		end tell
		display notification "Команда создана на рабочем столе" with title "OpenClaw Bridge"
	on error errMsg
		display dialog "Ошибка создания: " & errMsg with icon stop
	end try
end createQuickCommand
