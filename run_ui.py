import sys
#import jsonpickle
#jsonpickle.set_encoder_options('json', indent=4)
#jsonpickle.set_encoder_options('simplejson', indent=4)
import io
import os
from enum import Enum
import argparse

from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import * #QMainWindow, QDialog, QGraphicsScene, QListWidget, QListWidgetItem, QApplication, QSizePolicy
from PyQt6.QtWidgets import QListWidgetItem
from PyQt6 import uic

from core import ID, TournamentAction, Tournament, Player, Pod, Round, Log

class UILog:
    backlog = 0

    @classmethod
    def with_status(cls, func_to_decorate):
        def wrapper(self, *original_args, **original_kwargs):
            ret = func_to_decorate(self, *original_args, **original_kwargs)
            if cls.backlog < len(Log.output):
                for i in range(cls.backlog, len(Log.output)):
                    self.ui.lw_status.addItem(str(Log.output[i]))
                cls.backlog = len(Log.output)
                #TODO: Scroll
                #self.ui.lw_status.scroll
            #Log.log()
            return ret
        return wrapper

class SORT_METHOD(Enum):
    ID=0
    NAME=1
    PTS=2

class PlayerListItem(QListWidgetItem):
    SORT_METHOD = SORT_METHOD.ID
    INVERSE_SORT = False

    def __init__(self, player):
        self.player = player
        QListWidgetItem.__init__(self, str(player), parent=None)

    #@classmethod
    #def sort_method_str(cls):
    #    return '{} {}'.format(
    #        cls.SORT_METHOD.name,
    #        'ASC' if cls.INVERSE_SORT else 'DESC'
    #    )

    def __ge__ (self, other):
        if self.SORT_METHOD == SORT_METHOD.ID:
            b = self.player.id >= other.player.id
        elif self.SORT_METHOD == SORT_METHOD.NAME:
            b =  self.player.name >= other.player.name
        elif self.SORT_METHOD == SORT_METHOD.PTS:
            b =  self.player.points < other.player.points
        return b if not self.INVERSE_SORT else not b

    def __lt__ (self, other):
        if self.SORT_METHOD == SORT_METHOD.ID:
            b =  self.player.ID < other.player.ID
        elif self.SORT_METHOD == SORT_METHOD.NAME:
            b =  self.player.name < other.player.name
        elif self.SORT_METHOD == SORT_METHOD.PTS:
            b =  self.player.points >= other.player.points
        return b if not self.INVERSE_SORT else not b

    @classmethod
    def toggle_sort(cls):
        if not cls.INVERSE_SORT:
            cls.INVERSE_SORT = True

        elif cls.SORT_METHOD == SORT_METHOD.ID:
            cls.SORT_METHOD = SORT_METHOD.NAME
            cls.INVERSE_SORT = False
        elif cls.SORT_METHOD == SORT_METHOD.NAME:
            cls.SORT_METHOD = SORT_METHOD.PTS
            cls.INVERSE_SORT = False
        elif cls.SORT_METHOD == SORT_METHOD.PTS:
            cls.SORT_METHOD = SORT_METHOD.ID
            cls.INVERSE_SORT = False
        #print(cls.sort_method_str())

class MainWindow(QMainWindow):
    def __init__(self, core):
        self.file_name = None

        QMainWindow.__init__(self)
        self.core = Tournament()

        #Window code
        self.ui = uic.loadUi('./ui/MainWindow.ui')
        self.setCentralWidget(self.ui)
        #self.changeTitle()
        self.resize(900, 750)

        self.ui.pb_add_player.clicked.connect(
            lambda: self.add_player(self.ui.le_player_name.text()))
        self.ui.le_player_name.returnPressed.connect(
            lambda: self.add_player(self.ui.le_player_name.text()))

        self.ui.lv_players.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.lv_players.customContextMenuRequested.connect(
            self.lv_players_rightclick_menu)

        self.ui.pb_pods.clicked.connect(lambda: self.create_pods())


        self.ui.DEBUG1.clicked.connect(self.update_player_list)
        self.ui.DEBUG1.setText('update_player_list')

        self.ui.DEBUG2.clicked.connect(self.random_results)
        self.ui.DEBUG2.setText('random_results')

        self.ui.DEBUG3.clicked.connect(self.toggle_player_list_sorting)
        self.ui.DEBUG3.setText('toggle_player_list_sorting')

    def lv_players_rightclick_menu(self, position):
        #Popup menu
        pop_menu = QMenu()
        delete_player_action = QAction('Remove player', self)
        #rename_player_action = QAction('Rename player', self)
        #Check if it is on the item when you right-click, if it is not, delete and modify will not be displayed.
        if self.ui.lv_players.itemAt(position):
            pop_menu.addAction(delete_player_action)
        #    pop_menu.addAction(rename_player_action)

        delete_player_action.triggered.connect(self.lva_remove_player)
        #rename_player_action.triggered.connect(self.lva_rename_player)
        pop_menu.exec(self.ui.lv_players.mapToGlobal(position))

    def lva_remove_player(self):
        player = self.ui.lv_players.currentItem().data(Qt.ItemDataRole.UserRole)
        self.confirm(
            'Remove {}?'.format(player.name),
            'Confirm player removal'
        )
        self.remove_player(player)
        self.ui.lv_players.takeItem(self.ui.lv_players.currentRow())

    #Renaming player
    #TODO
    '''
    def lva_rename_player(self):
        curRow = self.ui.lv_players.currentRow()
        item = self.ui.lv_players.item(curRow)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.ui.lv_players.editItem(item)

        def try_rename_player(*nargs):
            player = self.ui.lv_players.currentItem().data(Qt.ItemDataRole.UserRole)
            self.core.rename_player(player.name, new_name)


        self.ui.lv_players.itemChanged.connect(lambda *x: try_rename_player(x))
    '''

    @UILog.with_status
    def add_player(self, player_name):
        #player_name = self.ui.le_player_name.text()
        players = self.core.add_player(player_name)
        if len(players) == 1:
            player = players[0]
            self.ui.le_player_name.clear()
            list_item = PlayerListItem(player)
            list_item.setData(Qt.ItemDataRole.UserRole, player)
            self.ui.lv_players.addItem(list_item)

    @UILog.with_status
    def remove_player(self, player_name):
        self.core.remove_player(player_name)

    @UILog.with_status
    def create_pods(self):
        self.core.make_pods()

    def confirm(self, message, title=''):
        reply = QMessageBox()
        #Force cancel left and OK right
        reply.setStyleSheet('*{button-layout: 3}')
        reply.setText(message)
        reply.setWindowTitle(title)
        reply.setStandardButtons(
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok)

        x = reply.exec()

        return x == QMessageBox.StandardButton.Ok

    def toggle_player_list_sorting(self):
        PlayerListItem.toggle_sort()
        self.update_player_list()

    def update_player_list(self):
        for row in range(self.ui.lv_players.count()):
            item = self.ui.lv_players.item(row)
            data = item.data(Qt.ItemDataRole.UserRole)
            item.setText(str(data))
        self.ui.lv_players.sortItems()

    def random_results(self):
        self.core.random_results()

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--players', dest='players', nargs='*')
parser.add_argument('-f', '--file', dest='file')
parser.add_argument('-i', '--input', dest='input', nargs='*')

subparsers = parser.add_subparsers()

if __name__ == '__main__':
    #TODO: args thing
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--players', dest='players', nargs='*')
    parser.add_argument('-f', '--file', dest='file')
    parser.add_argument('-i', '--input', dest='input', nargs='*')
    args, unknown = parser.parse_known_args()

    if args.players:
        for p in args.players:
            self.core.add_player(p)
    '''

    app = QApplication(sys.argv)

    window = MainWindow(None)
    window.show()

    app.exec()
    sys.exit(app.exit())
    #app.exec_()
    #sys.exit(app.exit())