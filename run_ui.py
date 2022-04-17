import argparse
#import jsonpickle
#jsonpickle.set_encoder_options('json', indent=4)
#jsonpickle.set_encoder_options('simplejson', indent=4)
import io
import os
import sys
from enum import Enum

import names
from PyQt6 import uic
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *  # QMainWindow, QDialog, QGraphicsScene, QListWidget, QListWidgetItem, QApplication, QSizePolicy
from PyQt6.QtWidgets import QListWidgetItem

from core import ID, Log, Player, Pod, Round, Tournament, TournamentAction, SORT_ORDER, SORT_METHOD

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
                self.ui.lw_status.scrollToBottom()
                self.ui.lw_status.setCurrentRow(self.ui.lw_status.count()-1)
            #Log.log()
            return ret
        return wrapper

class PlayerListItem(QListWidgetItem):
    def __init__(self, player:Player):
        QListWidgetItem.__init__(self, str(player), parent=None)
        self.player = player

    def __lt__(self, other):
        if isinstance(other, PlayerListItem):
            return self.player.__lt__(other.player)
        return False

    def __gt__(self, other):
        return self.player.__gt__(other.player)

    @staticmethod
    def toggle_sort():
        if Player.SORT_ORDER == SORT_ORDER.ASCENDING:
            Player.SORT_ORDER = SORT_ORDER.DESCENDING
        elif Player.SORT_METHOD == SORT_METHOD.ID:
            Player.SORT_METHOD = SORT_METHOD.NAME
            Player.SORT_ORDER = SORT_ORDER.ASCENDING
        elif Player.SORT_METHOD == SORT_METHOD.NAME:
            Player.SORT_METHOD = SORT_METHOD.RANK
            Player.SORT_ORDER = SORT_ORDER.ASCENDING
        elif Player.SORT_METHOD == SORT_METHOD.RANK:
            Player.SORT_METHOD = SORT_METHOD.ID
            Player.SORT_ORDER = SORT_ORDER.ASCENDING
        print(Player.SORT_METHOD.name, Player.SORT_ORDER.name)

    @staticmethod
    def SORT_ORDER():
        if Player.SORT_ORDER == SORT_ORDER.ASCENDING:
            return Qt.SortOrder.AscendingOrder
        return Qt.SortOrder.DescendingOrder

    #overwrite
    def text(self, tokens=['-i', '-p']):
        return self.player.__repr__(tokens)

class MainWindow(QMainWindow):
    def __init__(self, core=None):
        self.file_name = None

        QMainWindow.__init__(self)
        self.core = core if core else Tournament()

        #Window code
        self.ui = uic.loadUi('./ui/MainWindow.ui')
        self.setCentralWidget(self.ui)

        self.seated_color = QColor(0, 204, 102)
        self.unseated_color = QColor(117, 117, 163)
        #self.changeTitle()
        self.resize(900, 750)

        self.init_sort_dropdown()
        self.ui.cb_sort.currentIndexChanged.connect(self.cb_sort_set)

        self.ui.pb_add_player.clicked.connect(
            lambda: self.add_player(self.ui.le_player_name.text()))
        self.ui.le_player_name.returnPressed.connect(
            lambda: self.add_player(self.ui.le_player_name.text()))

        self.ui.lv_players.customContextMenuRequested.connect(
            self.lv_players_rightclick_menu)

        self.ui.pb_reset_pods.clicked.connect(lambda: self.reset_pods())
        self.ui.pb_pods.clicked.connect(lambda: self.create_pods())

        self.ui.actionReset_UI.triggered.connect(self.restore_ui)
        self.ui.actionRandom_Results.triggered.connect(lambda *_: self.random_results())

        self.ui.actionNew_tour.triggered.connect(self.new_tour)
        self.ui.actionLoad_state.triggered.connect(self.load_state)
        self.ui.actionLoad_tour.triggered.connect(self.load_tour)
        self.ui.actionSave_As.triggered.connect(self.save_as)

        self.ui.actionPods.triggered.connect(self.export_pods)
        self.ui.actionStandings.triggered.connect(self.export_standings)

        self.ui.actionLoad_players.triggered.connect(self.load_players)

        self.restore_ui()

    def load_players(self):
        file, ext = QFileDialog.getOpenFileName(
            caption='Select text file with players to load...',
            filter='*.txt',
            initialFilter='*.txt',
        )
        if file:
            with open(file, 'r') as f:
                player_names = f.readlines()
            self.core.add_player([p.strip() for p in player_names])
            self.restore_ui()

    def export_standings(self):
        file, ext = QFileDialog.getSaveFileName(
            caption="Specify standings location...",
            filter='*.txt',
            initialFilter='*.txt'
        )
        if file:
            if not file.endswith(ext.replace('*', '')):
                file = ext.replace('*', '{}').format(file)

            method = Player.SORT_METHOD
            order = Player.SORT_ORDER
            Player.SORT_METHOD = SORT_METHOD.RANK
            Player.SORT_ORDER = SORT_ORDER.DESCENDING
            players = sorted(self.core.players, reverse=True)
            Player.SORT_METHOD = method
            Player.SORT_ORDER = order

            maxlen = max([len(p.name) for p in players])

            standings = '\n'.join(
                [
                '[{:02d}] {} | {} | {:.2f} | {}'.format(
                    i+1,
                    p.name.ljust(maxlen),
                    p.points,
                    p.opponent_winrate,
                    p.unique_opponents
                )
                for i, p in zip(range(len(players)), players)
            ])

            self.core.export_str(file, standings)

    def export_pods(self):
        file, ext = QFileDialog.getSaveFileName(
            caption="Specify pods printout location...",
            filter='*.txt',
            initialFilter='*.txt'
        )
        if file:
            if not file.endswith(ext.replace('*', '')):
                file = ext.replace('*', '{}').format(file)

            pods_str = '\n\n'.join([
                pod.__repr__()
                for pod in self.core.round.pods
            ])

            self.core.export_str(file, pods_str)

    def init_sort_dropdown(self):
        values = [
            (SORT_METHOD.ID, SORT_ORDER.ASCENDING),
            (SORT_METHOD.ID, SORT_ORDER.DESCENDING),
            (SORT_METHOD.NAME, SORT_ORDER.ASCENDING),
            (SORT_METHOD.NAME, SORT_ORDER.DESCENDING),
            (SORT_METHOD.RANK, SORT_ORDER.ASCENDING),
            (SORT_METHOD.RANK, SORT_ORDER.DESCENDING),
        ]

        for tup in values:
            self.ui.cb_sort.addItem(
                '{} {}'.format(tup[0].name, tup[1].name),
                userData=tup
            )

    def load_state(self):
        state = LogLoaderDialog.show_dialog(self)
        if state:
            self.core = state
        self.restore_ui()

    def restore_ui(self):

        #clear pods
        self.ui_clear_pods()
        self.ui_create_pods()

        #clear players
        self.ui.lv_players.clear()
        self.ui_create_player_list()

    def lv_players_rightclick_menu(self, position):
        #Popup menu
        pop_menu = QMenu()
        manual_pod_action = QAction('Create pod', self)
        multiple = len(self.ui.lv_players.selectedItems()) > 1
        #Check if it is on the item when you right-click, if it is not, delete and modify will not be displayed.
        if self.ui.lv_players.itemAt(position):
            delete_player_action = QAction(
                'Remove player'
                if not multiple
                else 'Remove players',
                self
            )
            pop_menu.addAction(delete_player_action)
            delete_player_action.triggered.connect(self.lva_remove_player)

            if multiple:
                pop_menu.addAction(manual_pod_action)
                manual_pod_action.triggered.connect(self.lva_manual_pod)

        pop_menu.exec(self.ui.lv_players.mapToGlobal(position))

    #@UILog.with_status
    def lva_remove_player(self):
        players = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.ui.lv_players.selectedItems()
        ]
        #player = self.ui.lv_players.currentItem().data(Qt.ItemDataRole.UserRole)
        ok = self.confirm(
            'Remove {}?'.format(', '.join([p.name for p in players])),
            'Confirm player removal'
        )
        if ok:
            self.remove_player(players)
            self.ui.lv_players.clear()
            self.ui_create_player_list()

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

    def cb_sort_set(self, idx):
        method, order = self.ui.cb_sort.itemData(idx)
        Player.SORT_METHOD = method
        Player.SORT_ORDER = order
        self.ui.lv_players.sortItems(order=PlayerListItem.SORT_ORDER())

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
        self.ui_update_player_list()

    @UILog.with_status
    def remove_player(self, player_name):
        self.core.remove_player(player_name)
        self.ui_update_player_list()

    @UILog.with_status
    def create_pods(self):
        self.core.make_pods()
        self.ui_clear_pods()
        self.ui_create_pods()
        self.ui_update_player_list()

    @UILog.with_status
    def reset_pods(self):
        self.core.reset_pods()
        self.restore_ui()

    #@UILog.with_status
    def lva_manual_pod(self):
        self.core.manual_pod([
            p.data(Qt.ItemDataRole.UserRole)
            for p in self.ui.lv_players.selectedItems()
            if not p.data(Qt.ItemDataRole.UserRole).seated
        ])
        self.restore_ui()
        self.ui_update_player_list()

    def ui_create_pods(self):
        layout = self.ui.saw_content.layout()
        if self.core.round:
            for pod in self.core.round.pods:
                if not pod.done:
                    layout.addWidget(PodWidget(self, pod))

    def ui_clear_pods(self):
        layout = self.ui.saw_content.layout()
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

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

    def ui_toggle_player_list_sorting(self):
        PlayerListItem.toggle_sort()
        self.ui_update_player_list()

    def ui_update_player_list(self):
        for row in range(self.ui.lv_players.count()):
            item = self.ui.lv_players.item(row)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data.seated:
                item.setBackground(self.seated_color)
            else:
                item.setBackground(self.unseated_color)
            item.setText(str(data))
        self.ui.lv_players.sortItems(order=PlayerListItem.SORT_ORDER())

    def ui_create_player_list(self):
        for p in self.core.players:
            list_item = PlayerListItem(p)
            list_item.setData(Qt.ItemDataRole.UserRole, p)
            if p.seated:
                list_item.setBackground(self.seated_color)
            else:
                list_item.setBackground(self.unseated_color)
            self.ui.lv_players.addItem(list_item)
        self.ui.lv_players.sortItems(order=PlayerListItem.SORT_ORDER())

    @UILog.with_status
    def random_results(self):
        self.core.random_results()
        self.restore_ui()

    @UILog.with_status
    def report_win(self, player):
        Log.log('Reporting player {} won this round.'.format(player.name))
        self.core.report_win(player)
        self.ui_update_player_list()

    @UILog.with_status
    def report_draw(self, players):
        Log.log('Reporting draw for players: {}.'.format(
            ', '.join([p.name for p in players])
        ))
        self.core.report_draw(players)
        self.ui_update_player_list()

    def save_as(self):
        file, ext = QFileDialog.getSaveFileName(
            caption="Specify log location...",
            filter='*.log',
            initialFilter='*.log',
            directory=os.path.dirname(TournamentAction.DEFAULT_LOGF)
        )
        if file:
            if not file.endswith(ext.replace('*', '')):
                file = ext.replace('*', '{}').format(file)
            TournamentAction.LOGF = file
            TournamentAction.store()

    def new_tour(self):
        file, ext = QFileDialog.getSaveFileName(
            caption="Specify log location...",
            filter='*.log',
            initialFilter='*.log',
            directory=os.path.dirname(TournamentAction.DEFAULT_LOGF)
        )
        if file:
            if not file.endswith(ext.replace('*', '')):
                file = ext.replace('*', '{}').format(file)
            TournamentAction.LOGF = file
            self.core = Tournament()
            TournamentAction.ACTIONS=[]
            TournamentAction.store()
            self.restore_ui()

    def load_tour(self):
        file, ext = QFileDialog.getOpenFileName(
            caption='Select a log to open...',
            directory=os.path.dirname(TournamentAction.DEFAULT_LOGF),
            filter='*.log',
            initialFilter='*.log',
        )
        if file:
            TournamentAction.load(file)
            try:
                self.core = TournamentAction.ACTIONS[-1].after
            except Exception as e:
                self.core = Tournament()
            self.restore_ui()

class PodWidget(QWidget):
    def __init__(self, app, pod:Pod, parent=None):
        QWidget.__init__(self, parent=parent)
        self.app = app
        self.pod = pod
        self.ui = uic.loadUi('./ui/PodWidget.ui', self)
        self.refresh_ui()

        self.ui.lw_players.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        my_object = QObject()
        self.ui.lw_players.customContextMenuRequested.connect(
                self.rightclick_menu
        )

    def refresh_ui(self):
        self.ui.lbl_pod_id.setText(
            'Pod {} - {} players'.format(
                self.pod.id,
                len(self.pod.players)
            )
        )
        self.lw_players.clear()
        for p in self.pod.players:
            list_item = PlayerListItem(p)
            list_item.setData(Qt.ItemDataRole.UserRole, p)
            self.lw_players.addItem(list_item)

        self.lw_players.setFixedHeight(
            self.lw_players.sizeHintForRow(0) * self.lw_players.count() + 2 * self.lw_players.frameWidth()
        )

    def rightclick_menu(self, position):
        #Popup menu
        pop_menu = QMenu()
        report_win = QAction('Report win', self)
        report_draw = QAction('Report draw', self)
        #Check if it is on the item when you right-click, if it is not, delete and modify will not be displayed.
        if self.ui.lw_players.itemAt(position):
            if len(self.lw_players.selectedItems()) == 1:
                pop_menu.addAction(report_win)
            else:
                pop_menu.addAction(report_draw)

        report_win.triggered.connect(self.report_win)
        report_draw.triggered.connect(self.report_draw)
        #rename_player_action.triggered.connect(self.lva_rename_player)
        pop_menu.exec(self.ui.lw_players.mapToGlobal(position))

    def report_win(self, *nargs, **kwargs):
        player = self.lw_players.currentItem().data(Qt.ItemDataRole.UserRole)
        ok = self.app.confirm('Report player {} won?'.format(player.name), 'Confirm result')
        if ok:
            self.app.report_win(player)
            self.deleteLater()

    def report_draw(self):
        players = [
            item.data(Qt.ItemDataRole.UserRole)
            for item
            in self.lw_players.selectedItems()
        ]
        ok = self.app.confirm(
            'Report draw for players:\n\n{}'.format(
                '\n'.join([p.name for p in players])
            ),
            'Confirm result'
        )
        if ok:
            self.app.report_draw(players)
            self.deleteLater()

class LogLoaderDialog(QDialog):
    def __init__(self, app, parent=None):
        QDialog.__init__(self, parent)
        self.app = app

        self.ui = uic.loadUi('./ui/LogLoader.ui', self)
        self.restore_ui()

        self.pb_load_before.clicked.connect(lambda: self.load(True))
        self.pb_load_after.clicked.connect(lambda: self.load(False))
        self.pb_cancel.clicked.connect(lambda: self.done(1))

        self.action = None

    def restore_ui(self):
        self.lw_actions.clear()
        for action in TournamentAction.ACTIONS:
            item = QListWidgetItem('[{}] {}({}{})'.format(
                action.time.strftime('%H:%M'),
                action.func_name,
                ', '.join([str(arg) for arg in action.nargs]),
                ', '.join(['{}={}'.format(
                    key, val
                ) for key, val
                in action.kwargs.items()
                ])
            ))
            item.setData(Qt.ItemDataRole.UserRole, action)
            self.lw_actions.addItem(item)

    def load(self, before):
        action = self.lw_actions.currentItem().data(Qt.ItemDataRole.UserRole)
        if before:
            self.action = action.before
        else:
            self.action = action.after
        self.done(0)

    @staticmethod
    def show_dialog(app, parent=None):
        dlg = LogLoaderDialog(app, parent)
        dlg.show()
        result = dlg.exec()
        if result == 0:
            return dlg.action
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--players', dest='players', nargs='*')
    parser.add_argument('-f', '--file', dest='file')
    parser.add_argument('-i', '--input', dest='input', nargs='*')
    parser.add_argument('-n', '--number_of_mock_players', dest='number_of_mock_players', type=int, default=0)
    parser.add_argument('-s', '--pod_sizes', dest='pod_sizes', nargs='*', type=int, default=None)
    parser.add_argument('-b', '--allow_bye', dest='allow_bye', action='store_true', default=False)

    subparsers = parser.add_subparsers()
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)

    core = Tournament()
    if args.pod_sizes:
        Tournament.set_pod_sizes(args.pod_sizes)
    if args.allow_bye:
        Tournament.set_allow_bye(True)
    core.add_player([
        names.get_full_name()
        for i in range(args.number_of_mock_players)
    ])
    #for i in range(7):
    #   core.make_pods()
    #   core.random_results()

    window = MainWindow(core)
    window.show()

    app.exec()
    sys.exit(app.exit())
    #app.exec_()
    #sys.exit(app.exit())
