## V2 tasklist

* [ ] Implement GUI round selection

## Bugs
* [ ] Fix dropped players consistency (for log, tournament, round...)
* [x] Deepcopy crash on very large tournaments
* [x] Id generator doesn't get restored

## Misc improvements
* [ ] Discard default.log if mismatching file format
* [ ] Search function
* [ ] Model based item view for player list
* [ ] Window title should show tournament name, round, log path etc.
* [x] Automatic pairings and standings output
* [x] Clear/Update LOG on restore_ui
* [x] Fix B/W/L output in standings

## Discord integration
* [ ] Discord handle
* [ ] Push notifications
	* [ ] Start/end round
	* [ ] Start time of incoming round
	* [ ] Pairings
* [ ] Results submission

## Additional player information
* [ ] decklist submission
* [ ] Discord tags
* [ ] WOTC ID
* [ ] Optional commander input/parsed from moxfield

## Matchmaking improvements
* [ ] Implement a dynamic reward system (based on win/draw/loss rates which awards dynamic rewards for outcomes)
* [x] Improve bye awarding logic (only to bottom standings)
* [x] Seat normalization should not be uniform - 1+4 is not equal as 2+3
* [x] Develop a stricter logic for assigning seat order


## New functionality
* [ ] Implement top cut functionality
* [ ] Develop a league mode option
* [ ] Add git/versioning logic so that .log files containt data about commit (and appropriate commit can be loaded to inspect old data)
* [x] Export tournament log in a parsable (json?) format