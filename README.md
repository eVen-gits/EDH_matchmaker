# EDH_matchmaker #

## Supported commands in REPL: ##

* **add [player name]**; adds a player to the pool of available players
* **draw [p1, p2, ...]**; players that receive a point from a draw
* **h, help**; prints help (this file)
* **log**; prints the tournament log
* **pods**; prepares pods for the next round
* **print**; exports the last output to *print.txt* file
* **q**; quit
* **random**; reports random results for current round
* **remove [player name]**; removes player from the tournament
* **resetpods**; removes all pods so you can generate or manually create new ones
* **showpods**; prints out the pods
* **spods [n]**; calculates pod sizes for a given number of players
* **stats**; displays all player stats for all players. Supports formatting
	* use "stats -h" for more info
* **win [p1, p2, ...]**; reports a win for one or more players

## Important:

Player names are splitted on space - if you want to use first and last name, use either single or double quotes (add "John Doe" or 'Jane Doe').

## Installation:

To install required packages, use pip.
`pip3 install -r requirements.txt`
