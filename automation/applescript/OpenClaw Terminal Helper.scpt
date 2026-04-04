-- OpenClaw Terminal Helper
-- Интеграция OpenClaw с Terminal.app для выполнения команд

property openclawPath : "/Users/" & (do shell script "whoami") & "/.nvm/versions/node/v24.11.1/bin/openclaw"

-- Запуск команды в новом окне Terminal
on runInTerminal(command)
	tell application "Terminal"
		set newTab to do script "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && " & command
		activate
		return newTab
	end tell
end runInTerminal

-- Выполнение OpenClaw команды и возврат результата
on executeOpenClaw(subCommand)
	try
		set fullCmd to "export PATH=\"$HOME/.nvm/versions/node/v24.11.1/bin:$PATH\" && " & openclawPath & " " & subCommand
		set result to do shell script fullCmd
		return result
	on error errMsg
		return "Ошибка: " & errMsg
	end try
end executeOpenClaw

-- Основной обработчик
on run
	set commands to {"gateway", "status", "health", "agent --help", "dashboard", "tui"}
	set chosenCmd to choose from list commands with title "OpenClaw Terminal Helper" with prompt "Выберите команду для выполнения в Terminal:" OK button name "Выполнить" cancel button name "Отмена"
	
	if chosenCmd is false then return
	
	set cmd to item 1 of chosenCmd
	
	if cmd is "dashboard" or cmd is "tui" then
		-- Для интерактивных команд открываем Terminal
		runInTerminal("openclaw " & cmd)
	else
		-- Для остальных можно выполнить и показать результат
		set result to executeOpenClaw(cmd)
		if length of result > 500 then
			-- Если результат длинный, показываем в Terminal
			runInTerminal("openclaw " & cmd & " && echo 'Нажмите Enter для закрытия' && read")
		else
			display dialog "Результат:" & return & return & result buttons {"OK"} default button "OK"
		end if
	end if
end run
