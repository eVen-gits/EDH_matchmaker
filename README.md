# EDH_matchmaker #

## Supported commands in REPL: ##

* **add [player name]**; adds a player to the pool of available players
* **remove [player name]**; removes player from the tournament
* **stats**; displays all player stats for all players. Supports formatting
	* use "stats -h" for more info
* **pods**; prepares pods for the next round
* **random**; reports random results for current round
* **spods [n]**; calculates pod sizes for a given number of players
* **won [p1, p2, ...]**; reports a win for one or more players
* **draw [p1, p2, ...]**; players that receive a point from a draw
* **log**; prints the tournament log
* **print**; exports the last output to *print.txt* file
* **h, help**; prints help (this file)
* **q**; quit

## Important:

Player names are splitted on space - if you want to use first and last name, use either single or double quotes (add "John Doe" or 'Jane Doe').

## Installation:

To install required packages, use pip.
`pip3 install -r requirements.txt`