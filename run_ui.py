from __future__ import annotations

import argparse
import os
import sys


#import names
from faker import Faker
from PyQt6 import uic
from PyQt6.QtCore import *
from PyQt6.QtGui import *
# QMainWindow, QDialog, QGraphicsScene, QListWidget, QListWidgetItem, QApplication, QSizePolicy
from PyQt6.QtWidgets import *
from PyQt6.QtWidgets import QListWidgetItem

from src.interface import *
from src.core import (SortMethod, SortOrder, StandingsExport, Log, Player, Pod,
                  Tournament, TournamentAction, TournamentConfiguration)
from src.core import (StandingsExport, Log, Player, Pod,
                  Tournament, TournamentConfiguration, TournamentAction, TournamentContext)
from src.misc import generate_player_names


# from PySide2 import QtWidgets
# from PyQt5 import QtWidgets
#from qt_material import apply_stylesheet


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
            # Log.log()
            return ret
        return wrapper


class PlayerListItem(QListWidgetItem):
    class Color:
        SEATED = QColor(58, 170, 186)
        UNSEATED = QColor(12, 89, 100)

        WIN = QColor(24, 165, 85)
        BYE = QColor(121, 196, 140)
        DRAW = QColor(186, 84, 207)
        DEFEAT = QColor(241, 118, 120)
        GAMELOSS = QColor(219, 7, 61)

        DROP = QColor(128, 128, 128)

    def __init__(self, player: Player, p_fmt=None, parent=None, context: TournamentContext|None=None):
        if p_fmt is None:
            p_fmt = '-n -p -l'.split()
        QListWidgetItem.__init__(self, player.__repr__(tokens=p_fmt, context=context), parent=parent)

        monospace_font = QFont("Monospace")  # Use the generic "Monospace" font family
        self.setFont(monospace_font)
        self.player = player

    def __lt__(self, other: PlayerListItem):
        return bool(self.player.__lt__(other.player))

    def __gt__(self, other: PlayerListItem):
        return bool(self.player.__gt__(other.player))

    @staticmethod
    def sort_order():
        if Player.SORT_ORDER == SortOrder.ASCENDING:
            return Qt.SortOrder.AscendingOrder
        return Qt.SortOrder.DescendingOrder

    # overwrite
    def text(self, tokens=['-i', '-p']):
        return self.player.__repr__(tokens)


class GeneratePlayersDialog(QDialog):
    def __init__(self, parent=None):
        #simple dialog containing a spinbox and confirm/cancel buttons
        QDialog.__init__(self, parent)

        # Generate elements with code
        self.setWindowTitle("Generate players")
        self.m_layout = QVBoxLayout()
        self.setLayout(self.m_layout)

        self.sb_nPlayers = QSpinBox()
        self.sb_nPlayers.setRange(1, 1024)
        self.sb_nPlayers.setValue(16)
        self.m_layout.addWidget(self.sb_nPlayers)

        self.pb_confirm = QPushButton('Generate')
        self.pb_cancel = QPushButton('Cancel')
        self.m_layout.addWidget(self.pb_confirm)
        self.m_layout.addWidget(self.pb_cancel)

        self.pb_confirm.clicked.connect(self.generate)
        self.pb_cancel.clicked.connect(self.close)

    def generate(self):
        N = self.sb_nPlayers.value()
        new_names = generate_player_names(N)

        self.parent().core.add_player(list(new_names)) # pyright: ignore
        self.close()

    @staticmethod
    def show_dialog(parent):
        dlg = GeneratePlayersDialog(parent=parent)
        dlg.show()
        _ = dlg.exec()
        parent.restore_ui()
        return None


class MainWindow(QMainWindow):
    PLIST_FMT = '-n -p -l'.split()

    def __init__(self, core: Tournament):
        self.file_name = None

        QMainWindow.__init__(self)
        self.core = core if core else Tournament()

        '''TODO: Custom theme
        extra = {

            # Button colors
            'danger': '#dc3545',
            'warning': '#ffc107',
            'success': '#17a2b8',

            # Font
            'font_family': 'monoespace',
            'font_size': '13px',
            'line_height': '13px',

            # Density Scale
            'density_scale': '-3',

            # environ
            'pyside6': True,
            'linux': True,

        }

        apply_stylesheet(app, theme='dark_amber.xml', extra=extra, css_file='ui/custom_theme.css')
        '''

        self.setWindowTitle("EDH matchmaker")

        # Window code
        self.ui = uic.loadUi('./ui/MainWindow.ui')
        self.setCentralWidget(self.ui)

        # self.changeTitle()
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
        self.ui.actionRandom_Results.triggered.connect(
            lambda *_: self.random_results())
        self.ui.actionGenerate_Players.triggered.connect(
            lambda: GeneratePlayersDialog.show_dialog(self)
        )


        self.ui.actionNew_tour.triggered.connect(self.new_tour)
        self.ui.actionTour_config.triggered.connect(self.edit_tour)
        self.ui.actionSelect_Round.triggered.connect(self.select_round)
        self.ui.actionLoad_tour.triggered.connect(self.load_tour)
        self.ui.actionSave_As.triggered.connect(self.save_as)

        self.ui.actionPodsDiscord.triggered.connect(self.export_pods_discord)
        self.ui.actionPodsOnline.triggered.connect(self.export_pods_online)
        self.ui.actionPods.triggered.connect(self.export_pods)
        self.ui.actionStandings.triggered.connect(self.export_standings_dialog)

        self.ui.actionLoad_players.triggered.connect(self.load_players)

        #self.ui.actionJSON_Log.triggered.connect(lambda: print(self.core.parsable_log())) #TODO: Reimplement

        #self.ui.actionUndo.triggered.connect(self.undo)
        #self.ui.actionRedo.triggered.connect(self.redo)

        self.restore_ui()

        icon = QIcon('media/icon.ico')
        self.setWindowIcon(icon)

    def load_players(self):
        file, ext = QFileDialog.getOpenFileName(
            caption='Select text file with players to load...',
            filter='*.txt',
            initialFilter='*.txt',
        )
        if file:
            with open(file, 'r', encoding='utf-8') as f:
                player_names = f.readlines()
            self.core.add_player([p.strip() for p in player_names])
            self.restore_ui()

    def export_standings_dialog(self):
        ExportStandingsDialog.show_dialog(self)

    def export_pods_discord(self):
        if self.core.tour_round:
            tour_round = self.core.tour_round
            self.core.export_str(
                f"# Round {tour_round.seq}\n",
                None,
                StandingsExport.Target.DISCORD
            )
        else:
            tour_round = self.core.rounds[-1]
            self.core.export_str(
                f"# Round {tour_round.seq} results\n",
                None,
                StandingsExport.Target.DISCORD
            )

        for pod in tour_round.pods:
            try:
                self.core.export_str(
                    f"```\n{pod.__repr__()}\n```\n",
                    None,
                    StandingsExport.Target.DISCORD
                )
            except Exception as e:
                pass

    def export_pods_online(self):
        if self.core.tour_round:
            try:
                self.core.export_str(
                    self.core.get_pods_str(),
                    None,
                    StandingsExport.Target.WEB
                )
            except Exception as e:
                pass

    def export_pods(self):
        if self.core.tour_round:
            file, ext = QFileDialog.getSaveFileName(
                caption="Specify pods printout location...",
                filter='*.txt',
                initialFilter='*.txt'
            )
            if file:
                if not file.endswith(ext.replace('*', '')):
                    file = ext.replace('*', '{}').format(file)
                pods_str = self.core.get_pods_str()
                self.core.export_str(pods_str, file, StandingsExport.Target.FILE)

    def init_sort_dropdown(self):
        values = [
            (SortMethod.RANK, SortOrder.ASCENDING),
            (SortMethod.RANK, SortOrder.DESCENDING),
            (SortMethod.ID, SortOrder.ASCENDING),
            (SortMethod.ID, SortOrder.DESCENDING),
            (SortMethod.NAME, SortOrder.ASCENDING),
            (SortMethod.NAME, SortOrder.DESCENDING),
        ]

        for tup in values:
            self.ui.cb_sort.addItem(
                '{} {}'.format(tup[0].name, tup[1].name),
                userData=tup
            )

    def select_round(self):
        selected_round = RoundSelectWidget.show_dialog(self)
        if selected_round:
            self.core.tour_round = selected_round
            self.restore_ui()

    #def undo(self):
    #    self.core = TournamentAction.ACTIONS[-1].before
    #    self.restore_ui()

    #def redo(self):
    #    self.core = TournamentAction.ACTIONS[-1].after
    #    self.restore_ui()

    def restore_ui(self):

        # clear pods
        self.ui_clear_pods()
        self.ui_create_pods()

        # clear players
        self.ui.lv_players.clear()
        self.ui_create_player_list()

    def lv_players_rightclick_menu(self, position):
        # Popup menu
        pop_menu = QMenu()
        multiple = len(self.ui.lv_players.selectedItems()) > 1
        # Check if it is on the item when you right-click, if it is not, delete and modify will not be displayed.
        item = self.ui.lv_players.itemAt(position)
        if item:
            if item.player in self.core.dropped:
                return
            delete_player_action = QAction(
                'Drop player'
                if not multiple
                else 'Drop players',
                self
            )
            pop_menu.addAction(delete_player_action)
            delete_player_action.triggered.connect(
                lambda: self.lva_remove_player())

            # TODO: Rename player option
            #rename_player_action = QAction('Rename player')
            # pop_menu.addAction(rename_player_action)
            # rename_player_action.triggered.connect(self.lva_rename_player)

            if self.core.tour_round:
                if self.core.tour_round.pods:
                    add_to_pod_action = QMenu('Move to pod')
                    pop_menu.addMenu(add_to_pod_action)
                    for pod in self.core.tour_round.pods:
                        add_to_pod_action.addAction(
                            pod.name,
                            #lambda pod=pod: self.lva_add_to_pod(pod)
                            lambda x=pod: self.lva_move_to_pod(x)
                        )
                    # add_to_pod_action.addAction(add_to_pod_action)
                    # add_to_pod_action.triggered.connect(self.lva_add_to_pod)

            if multiple:
                pop_menu.addAction(QAction(
                    'Create pod',
                    self,
                    triggered=lambda: self.lva_manual_pod()
                )) # type: ignore
            pop_menu.addAction(QAction(
                'Toggle game loss',
                self,
                triggered=lambda: self.lva_game_loss()
            )) # type: ignore
            pop_menu.addAction(QAction(
                'Toggle bye',
                self,
                triggered=lambda: self.lva_bye()
            )) # type: ignore
        pop_menu.exec(self.ui.lv_players.mapToGlobal(position))

    def lva_remove_player(self):
        players = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.ui.lv_players.selectedItems()
        ]
        #player = self.ui.lv_players.currentItem().data(Qt.ItemDataRole.UserRole)
        ok = self.confirm(
            'Drop {}?'.format(', '.join([p.name for p in players])),
            'Confirm player removal'
        )
        if ok:
            self.drop_player(players)
            self.ui.lv_players.clear()
            #self.ui_create_player_list()
            self.restore_ui()

    def lva_rename_player(self):
        # TODO:
        raise NotImplementedError()
        curRow = self.ui.lv_players.currentRow()
        item = self.ui.lv_players.item(curRow)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        item.setText(item.player.name)

    def lva_move_to_pod(self, pod: Pod):
        self.move_players_to_pod(
            pod,
            [
                item.player
                for item
                in self.ui.lv_players.selectedItems()
            ]
        )

    def lva_game_loss(self):
        players = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.ui.lv_players.selectedItems()
        ]
        ok = self.confirm(
            'Toggle game loss for: {}?'.format(
                ', '.join([p.name for p in players])),
            'Confirm game loss status'
        )
        if ok:
            self.toggle_game_loss(players)

    def lva_bye(self):
        players = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.ui.lv_players.selectedItems()
        ]
        ok = self.confirm(
            'Toggle bye for: {}?'.format(
                ', '.join([p.name for p in players])),
            'Confirm bye status'
        )
        if ok:
            self.toggle_bye(players)

    @UILog.with_status
    def toggle_game_loss(self, players: list[Player]):
        if not isinstance(players, list):
            players = [players]

        self.core.toggle_game_loss(players)
        self.ui_update_player_list()
        self.ui_update_pods()

    @UILog.with_status
    def toggle_bye(self, players: list[Player]):
        if not isinstance(players, list):
            players = [players]

        self.core.toggle_bye(players)
        self.ui_update_player_list()
        self.ui_update_pods()

    @UILog.with_status
    def move_players_to_pod(self, pod: Pod, players: list[Player]):
        self.core.move_player_to_pod(
            pod,
            players,
            manual=True)

        #Log.log('Added {} to pod {}'.format(item.player.name, pod.id))
        self.ui_update_player_list()
        self.ui_update_pods()

    def cb_sort_set(self, idx):
        self.update_player_sort_method()
        self.ui.lv_players.sortItems(order=PlayerListItem.sort_order()) # pyright: ignore
        pass

    @UILog.with_status
    def add_player(self, player_name):
        #player_name = self.ui.le_player_name.text()
        players = self.core.add_player(player_name)
        if len(players) == 1:
            player = players[0]
            self.ui.le_player_name.clear()
            list_item = PlayerListItem(player, p_fmt=self.PLIST_FMT)
            list_item.setData(Qt.ItemDataRole.UserRole, player)
            self.ui.lv_players.addItem(list_item)
        self.ui_update_player_list()

    @UILog.with_status
    def drop_player(self, player_name):
        self.core.drop_player(player_name)
        self.ui_update_player_list()

    @UILog.with_status
    def create_pods(self):
        if self.core.create_pairings():
            self.ui_clear_pods()
            self.ui_create_pods()
            self.ui_update_player_list()
            if self.core.tour_round.seq > 0:
                Log.log("Number of repeat pairings per pod:")
                repeat_pairings = self.core.tour_round.repeat_pairings()
                for i, pod in enumerate(self.core.tour_round.pods):
                    Log.log(f"Pod {i}: {repeat_pairings[pod]}")


    @UILog.with_status
    def reset_pods(self):
        self.core.reset_pods()
        self.restore_ui()

    @UILog.with_status
    def lva_manual_pod(self):
        players = [
            p.data(Qt.ItemDataRole.UserRole)
            for p in self.ui.lv_players.selectedItems()
            if not p.data(Qt.ItemDataRole.UserRole).seated(self.core.tour_round)
        ]
        self.core.manual_pod(players)
        self.restore_ui()
        self.ui_update_player_list()

    def ui_create_pods(self):
        layout = self.ui.saw_content.layout()
        if self.core.tour_round:
            context = TournamentContext(self.core, self.core.tour_round, self.core.get_standings(self.core.tour_round))
            for pod in self.core.tour_round.pods:
                layout.addWidget(PodWidget(self, pod, context=context))

    def ui_clear_pods(self):
        layout = self.ui.saw_content.layout()
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def ui_update_pods(self):
        layout = self.ui.saw_content.layout()
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            widget.refresh_ui()

    def confirm(self, message, title=''):
        reply = QMessageBox()
        # Force cancel left and OK right
        reply.setStyleSheet('*{button-layout: 3}')
        reply.setText(message)
        reply.setWindowTitle(title)
        reply.setStandardButtons(
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok)

        x = reply.exec()

        return x == QMessageBox.StandardButton.Ok

    def set_list_item_color(self, item: QListWidgetItem, context: TournamentContext):
        player: Player = item.data(Qt.ItemDataRole.UserRole)
        result = player.result(self.core.tour_round)

        if player in self.core.dropped:
            item.setBackground(PlayerListItem.Color.DROP)
        elif player.seated(self.core.tour_round):
            if result == Player.EResult.PENDING:
                item.setBackground(PlayerListItem.Color.SEATED)
            elif result == Player.EResult.LOSS:
                if player in context.tour_round.game_loss:
                    item.setBackground(PlayerListItem.Color.GAMELOSS)
                else:
                    item.setBackground(PlayerListItem.Color.DEFEAT)
            elif result == Player.EResult.WIN:
                item.setBackground(PlayerListItem.Color.WIN)
            elif result == Player.EResult.DRAW:
                item.setBackground(PlayerListItem.Color.DRAW)
        else:
            if player in context.tour_round.game_loss:
                item.setBackground(PlayerListItem.Color.GAMELOSS)
            elif player in context.tour_round.byes:
                item.setBackground(PlayerListItem.Color.BYE)
            else:
                item.setBackground(PlayerListItem.Color.UNSEATED)

    def ui_update_player_list(self):
        context = TournamentContext(self.core, self.core.tour_round, self.core.get_standings(self.core.tour_round))
        for row in range(self.ui.lv_players.count()):
            item = self.ui.lv_players.item(row)
            data: Player = item.data(Qt.ItemDataRole.UserRole)

            self.set_list_item_color(item, context)

            item.setText(data.__repr__(self.PLIST_FMT, context=context))
        self.ui.lv_players.sortItems(order=PlayerListItem.sort_order())

    def update_player_sort_method(self):
        sort_method, sort_order = self.ui.cb_sort.itemData(self.ui.cb_sort.currentIndex()) # type: ignore
        if Player.SORT_METHOD != sort_method:
            Player.SORT_METHOD = sort_method
        if Player.SORT_ORDER != sort_order:
            Player.SORT_ORDER = sort_order

    def ui_create_player_list(self):
        tour_round = self.core.tour_round
        standings = self.core.get_standings(tour_round)
        context = TournamentContext(self.core, tour_round, standings)
        for p in self.core.players:
            list_item = PlayerListItem(p, p_fmt=self.PLIST_FMT, context=context)
            list_item.setData(Qt.ItemDataRole.UserRole, p)
            self.set_list_item_color(list_item, context)
            self.ui.lv_players.addItem(list_item)
        self.update_player_sort_method()
        self.ui.lv_players.sortItems(order=PlayerListItem.sort_order()) # type: ignore

    @UILog.with_status
    def random_results(self):
        self.core.random_results()
        self.restore_ui()

    @UILog.with_status
    def report_win(self, player: Player):
        self.core.report_win(player)
        self.ui_update_player_list()

    @UILog.with_status
    def report_draw(self, players: list[Player]):
        Log.log('Reporting draw for players: {}.'.format(
            ', '.join([p.name for p in players])
        ))
        self.core.report_draw(players)
        self.ui_update_player_list()

    @UILog.with_status
    def bench_players(self, players: list[Player]):
        Log.log('Bench players: {}.'.format(
            ', '.join([p.name for p in players])
        ))
        self.core.bench_players(players)
        self.ui_update_player_list()

    @UILog.with_status
    def delete_pod(self, pod: Pod):
        self.core.delete_pod(pod)
        self.ui_update_player_list()

    def save_as(self):
        file, ext = QFileDialog.getSaveFileName(
            caption="Specify log location...",
            filter='*.json',
            initialFilter='*.json',
            directory=os.path.dirname(TournamentAction.DEFAULT_LOGF)
        )
        if file:
            if not file.endswith(ext.replace('*', '')):
                file = ext.replace('*', '{}').format(file)
            TournamentAction.LOGF = file
            TournamentAction.store()

    def new_tour(self):
        TournamentConfigDialog.show_dialog(self)
        self.restore_ui()

    def edit_tour(self):
        TournamentConfigDialog.show_edit_dialog(self)
        self.restore_ui()

    def load_tour(self):
        file, ext = QFileDialog.getOpenFileName(
            caption='Select a log to open...',
            directory=os.path.dirname(TournamentAction.DEFAULT_LOGF),
            filter='*.json',
            initialFilter='*.json',
        )
        if file:
            TournamentAction.load(file)
            try:
                self.core = TournamentAction.ACTIONS[-1].after
            except Exception as e:
                self.core = Tournament()
            self.restore_ui()


class PodWidget(QWidget):
    class Color:
        WIN = QColor(24, 165, 85)
        DEFEAT = QColor(241, 118, 120)
        GAMELOSS = QColor(219, 7, 61)
        DRAW = QColor(186, 84, 207)
        DROP = QColor(128, 128, 128)

    PLIST_FMT = '-n -p'.split()
    def __init__(self, app: MainWindow, pod: Pod, parent=None, context: TournamentContext|None=None):
        QWidget.__init__(self, parent=parent)
        self.app = app
        self.pod = pod
        self.ui = uic.loadUi('./ui/PodWidget.ui', self)
        self.refresh_ui(context=context)

        self.ui.lw_players.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.lw_players.customContextMenuRequested.connect(
            self.rightclick_menu
        )

    def refresh_ui(self, context: TournamentContext|None=None):
        if not self.pod.players:
            self.setParent(None)
            return

        if context is None:
            context = TournamentContext(
                self.app.core,
                self.pod.tour_round,
                self.app.core.get_standings(self.pod.tour_round)
            )

        self.ui.lbl_pod_id.setText(
            '{} - {} players'.format(
                self.pod.name,
                len(self.pod.players)
            )
        )
        self.lw_players.clear()
        for p in self.pod.players:
            list_item = PlayerListItem(p, p_fmt=self.PLIST_FMT, context=context)
            list_item.setData(Qt.ItemDataRole.UserRole, p)
            self.lw_players.addItem(list_item)
            if self.pod.done:
                if p in context.tour_round.game_loss:
                    list_item.setBackground(PodWidget.Color.GAMELOSS)
                elif p in self.pod.result:
                    if self.pod.result_type == Pod.EResult.DRAW:
                        list_item.setBackground(PodWidget.Color.DRAW)
                    elif self.pod.result_type == Pod.EResult.WIN:
                        list_item.setBackground(PodWidget.Color.WIN)
                else:
                    list_item.setBackground(PodWidget.Color.DEFEAT)
                    #    list_item.setBackground(PlayerListItem.Color.pod_loss)
                #if p in self.pod.tour_round.game_loss:
                #    list_item.setBackground(PlayerListItem.Color.pod_loss)

        self.lw_players.setFixedHeight(
            self.lw_players.sizeHintForRow(
                0) * self.lw_players.count() + 2 * self.lw_players.frameWidth()
        )

    def rightclick_menu(self, position):
        # Popup menu
        pop_menu = QMenu()

        selected_pod_players = [
            i.player for i in self.lw_players.selectedItems()
        ]
        n_selected = len(selected_pod_players)

        # Check if it is on the item when you right-click, if it is not, delete and modify will not be displayed.
        if self.ui.lw_players.itemAt(position):
            if n_selected == 1:
                pop_menu.addAction(
                    QAction('Report win', self, triggered=self.report_win))
            else:
                pop_menu.addAction(
                    QAction('Report draw', self, triggered=self.report_draw))

        move_pod = QMenu('Move to pod')
        pop_menu.addMenu(move_pod)
        for pod in self.app.core.tour_round.pods:
            if pod != self.pod:
                move_pod.addAction(pod.name,
                                   lambda pod=pod, players=selected_pod_players:
                                   self.app.move_players_to_pod(
                                       pod, players
                                   )
                                   )

        pop_menu.addAction(QAction(
            'Bench player'
            if n_selected == 1
            else 'Bench players',
            self,
            triggered=self.bench_players

        )) # type: ignore
        pop_menu.addAction(QAction(
            'Toggle game loss',
            self,
            triggered=self.toggle_game_loss
        )) # type: ignore
        pop_menu.addAction(QAction(
            'Assign bye',
            self,
            triggered=self.assign_bye
        )) # type: ignore
        pop_menu.addSeparator()
        pop_menu.addAction(
            QAction(
                'Delete pod',
                self,
                triggered=self.delete_pod
            )) # type: ignore
        pop_menu.addAction(
            QAction(
                'Reset result',
                self,
                triggered=self.reset_result
            )) # type: ignore

        # rename_player_action.triggered.connect(self.lva_rename_player)
        pop_menu.exec(self.ui.lw_players.mapToGlobal(position))

    def report_win(self):
        player = self.lw_players.currentItem().data(Qt.ItemDataRole.UserRole)
        ok = self.app.confirm('Report player {} won?'.format(
            player.name), 'Confirm result')
        if ok:
            self.app.report_win(player)
            #self.deleteLater()
        #context = TournamentContext(self.app.core, self.pod.tour_round, self.app.core.get_standings(self.pod.tour_round))
        self.refresh_ui()

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
            self.pod.reset_result()
            self.app.report_draw(players)
            #self.deleteLater()
            self.app.ui_update_player_list()
            self.refresh_ui()

    def reset_result(self):
        if self.pod.result_type != Pod.EResult.PENDING:
            self.pod.reset_result()
            self.app.ui_update_player_list()
            self.refresh_ui()

    def bench_players(self):
        players = [
            item.data(Qt.ItemDataRole.UserRole)
            for item
            in self.lw_players.selectedItems()
        ]
        if len(players) == self.pod._players:
            self.deleteLater()
        self.app.bench_players(players)
        self.refresh_ui()

    def delete_pod(self):
        self.app.delete_pod(self.pod)
        self.refresh_ui()

    def toggle_game_loss(self):
        players = [
            item.data(Qt.ItemDataRole.UserRole)
            for item
            in self.lw_players.selectedItems()
        ]
        ok = self.app.confirm(
            'Assign game loss for players:\n\n{}'.format(
                '\n'.join([p.name for p in players])
            ),
            'Confirm game loss penalty'
        )
        if ok:
            self.app.toggle_game_loss(players)

    def assign_bye(self):
        players = [
            item.data(Qt.ItemDataRole.UserRole)
            for item
            in self.lw_players.selectedItems()
        ]
        ok = self.app.confirm(
            'Assign bye for players:\n\n{}'.format(
                '\n'.join([p.name for p in players])
            ),
            'Confirm bye assignment'
        )
        if ok:
            self.app.toggle_bye(players)


class RoundSelectWidget(QDialog):
    def __init__(self, core: Tournament, parent=None):
        QDialog.__init__(self, parent)

        self.core = core

        self.ui = uic.loadUi('./ui/RoundSelectWidget.ui', self)
        assert self.ui is not None
        self.restore_ui()

    def restore_ui(self):
        for tour_round in self.core.rounds:
            item = QListWidgetItem(f'Round {tour_round.seq}')
            item.setData(Qt.ItemDataRole.UserRole, tour_round)
            self.ui.lw_rounds.addItem(item)

        self.ui.pb_confirm.clicked.connect(lambda: self.done(QDialog.DialogCode.Accepted))
        self.ui.pb_cancel.clicked.connect(lambda: self.done(QDialog.DialogCode.Rejected))

    @staticmethod
    def show_dialog(parent=None):
        dlg = RoundSelectWidget(parent.core)
        dlg.show()
        result = dlg.exec()
        if result:
            selected_round = dlg.ui.lw_rounds.currentItem().data(Qt.ItemDataRole.UserRole)
            return selected_round
        return None


class TournamentConfigDialog(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.core: Tournament = parent.core

        self.reset = True

        self.ui = uic.loadUi('./ui/TournamentConfigDialog.ui', self)
        assert self.ui is not None

        self.ui.cb_allow_bye.stateChanged.connect(self.ui.sb_bye.setEnabled)
        self.ui.cb_allow_bye.stateChanged.connect(self.ui.sb_max_byes.setEnabled)
        self.ui.pb_browse.clicked.connect(self.select_log_location)
        self.ui.pb_add_psize.clicked.connect(self.add_psize)
        self.ui.pb_remove_psize.clicked.connect(self.remove_psize)
        self.ui.pb_confirm.clicked.connect(self.apply_choices)
        self.ui.lw_pod_sizes.itemChanged.connect(self.check_pod_sizes)

        self.restore_ui()

    def restore_ui(self):
        # Load current pod sizes
        for psize in self.core.config.pod_sizes:
            self.create_psize_widget(psize)
        # Load and set bye option
        self.cb_allow_bye.setChecked(self.core.config.allow_bye)
        self.check_pod_sizes()
        # Load and set scoring
        self.sb_win.setValue(self.core.config.win_points)
        self.sb_draw.setValue(self.core.config.draw_points)
        self.sb_bye.setValue(self.core.config.bye_points)
        self.sb_nRounds.setValue(self.core.config.n_rounds)
        self.cb_snakePods.setChecked(self.core.config.snake_pods)
        self.sb_max_byes.setValue(self.core.config.max_byes)
        self.ui.cb_auto_export.setChecked(self.core.config.auto_export)
        # Populte cb_topCut
        self.ui.cb_topCut.addItem('None', TournamentConfiguration.TopCut.NONE)
        self.ui.cb_topCut.addItem('Top 4', TournamentConfiguration.TopCut.TOP_4)
        self.ui.cb_topCut.addItem('Top 7', TournamentConfiguration.TopCut.TOP_7)
        self.ui.cb_topCut.addItem('Top 10', TournamentConfiguration.TopCut.TOP_10)
        self.ui.cb_topCut.addItem('Top 13', TournamentConfiguration.TopCut.TOP_13)
        self.ui.cb_topCut.addItem('Top 16', TournamentConfiguration.TopCut.TOP_16)
        self.ui.cb_topCut.addItem('Top 40', TournamentConfiguration.TopCut.TOP_40)
        self.ui.cb_topCut.setCurrentIndex(self.ui.cb_topCut.findData(self.core.config.top_cut))

        if TournamentAction.LOGF:
            self.ui.le_log_location.setText(TournamentAction.LOGF)
        else:
            self.ui.le_log_location.setText(
                os.path.abspath(TournamentAction.DEFAULT_LOGF))

    def check_pod_sizes(self):
        items = [
            self.lw_pod_sizes.item(i)
            for i
            in range(self.lw_pod_sizes.count())
        ]
        for item in items:
            try:
                item.setData(Qt.ItemDataRole.UserRole, int(item.text()))
            except ValueError:
                self.lw_pod_sizes.takeItem(self.lw_pod_sizes.row(item))

    def remove_psize(self):
        if self.lw_pod_sizes.currentItem():
            self.lw_pod_sizes.takeItem(
                self.lw_pod_sizes.row(self.lw_pod_sizes.currentItem()))

    def add_psize(self):
        self.check_pod_sizes()
        current_values = [
            int(self.lw_pod_sizes.item(i).text())
            for i
            in range(self.lw_pod_sizes.count())
        ]
        if len(current_values) == 0:
            self.create_psize_widget(4)
        else:
            self.create_psize_widget(
                min(current_values) - 1
                if min(current_values) > 1
                else max(current_values) + 1
            )
        self.check_pod_sizes()

    def create_psize_widget(self, psize: int):
        item = QListWidgetItem(str(psize))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        item.setData(Qt.ItemDataRole.UserRole, psize)
        self.lw_pod_sizes.addItem(item)

    def get_psizes(self):
        return [
            self.lw_pod_sizes.item(i).data(Qt.ItemDataRole.UserRole)
            for i
            in range(self.lw_pod_sizes.count())
        ]

    def select_log_location(self):
        file, ext = QFileDialog.getSaveFileName(
            caption="Specify log location...",
            filter='*.json',
            initialFilter='*.json',
            directory=os.path.dirname(TournamentAction.DEFAULT_LOGF)
        )
        if file:
            if not file.endswith(ext.replace('*', '')):
                file = ext.replace('*', '{}').format(file)
            self.ui.le_log_location.setText(file)

    def apply_choices(self):
        TournamentAction.LOGF = self.ui.le_log_location.text()
        self.config = TournamentConfiguration(
            allow_bye = self.cb_allow_bye.isChecked(),
            win_points = self.sb_win.value(),
            draw_points = self.sb_draw.value(),
            bye_points = self.sb_bye.value(),
            pod_sizes = self.get_psizes(),
            n_rounds = self.sb_nRounds.value(),
            snake_pods = self.cb_snakePods.isChecked(),
            max_byes = self.sb_max_byes.value(),
            auto_export = self.cb_auto_export.isChecked(),
            top_cut = self.ui.cb_topCut.currentData(),
        )
        if self.reset:
            t = Tournament(
                config=self.config,
            )
            self.parent().core = t
            t.initialize_round()
            TournamentAction.store(t)
        else:
            self.parent().core.config = self.config

        self.close()

    @staticmethod
    def show_dialog(parent=None):
        dlg = TournamentConfigDialog(parent)
        dlg.show()
        result = dlg.exec()

    @staticmethod
    def show_edit_dialog(parent=None):
        dlg = TournamentConfigDialog(parent)
        dlg.reset = False
        dlg.show()
        result = dlg.exec()


class ExportStandingsDialog(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.core:Tournament = parent.core
        self.ui = uic.loadUi('./ui/ExportDialog.ui', self)
        assert self.ui is not None

        self.restore_ui()

        self.ui.pb_browse.clicked.connect(self.select_export_path)
        self.ui.cb_format.currentIndexChanged.connect(
            self.update_export_format)
        self.ui.pb_add.clicked.connect(self.add_field)
        self.ui.pb_remove.clicked.connect(self.remove_field)
        self.ui.pb_export.clicked.connect(self.export)

    def update_export_format(self, idx):
        data = self.ui.cb_format.itemData(idx)
        #StandingsExport.instance().format = data
        self.core.config.standings_export.format = data

    def restore_ui(self):
        self.ui.le_export_dir.setText(self.core.config.standings_export.dir)

        for f in StandingsExport.Field:
            info = self.core.config.standings_export.info[f]
            if f in self.core.config.standings_export.fields:
                item = QListWidgetItem(
                    '{} ({})'.format(info.name, info.description))
                item.setData(Qt.ItemDataRole.UserRole, f)
                self.ui.lw_fields.addItem(item)
            else:
                self.ui.cb_fields.addItem(info.name, userData=f)

        for s in StandingsExport.Format:
            self.ui.cb_format.addItem(s.name, userData=s)

        self.ui.cb_format.setCurrentIndex(
            self.ui.cb_format.findData(self.core.config.standings_export.format))

    def add_field(self):
        f = self.ui.cb_fields.currentData(Qt.ItemDataRole.UserRole)
        if f is None:
            return
        info = self.core.config.standings_export.info[f]
        self.ui.cb_fields.removeItem(self.ui.cb_fields.currentIndex())
        item = QListWidgetItem('{} ({})'.format(info.name, info.description))
        item.setData(Qt.ItemDataRole.UserRole, f)
        self.ui.lw_fields.addItem(item)

    def remove_field(self):
        item = self.ui.lw_fields.currentItem()
        f = item.data(Qt.ItemDataRole.UserRole)
        info = self.core.config.standings_export.info[f]
        self.ui.cb_fields.addItem(info.name, userData=f)
        self.ui.lw_fields.takeItem(self.ui.lw_fields.row(item))

    def select_export_path(self):
        file, ext = QFileDialog.getSaveFileName(
            caption="Specify standings location...",
            filter='*{}'.format(
                self.core.config.standings_export.ext[self.core.config.standings_export.format]),
            initialFilter='*{}'.format(
                self.core.config.standings_export.ext[self.core.config.standings_export.format]),
            directory=os.path.dirname(self.core.config.standings_export.dir)
        )
        if file:
            if not file.endswith(ext.replace('*', '')):
                file = ext.replace('*', '{}').format(file)
            self.ui.le_export_dir.setText(file)
            self.core.config.standings_export.dir = file

    def export(self):
        self.core.config.standings_export.format = self.ui.cb_format.currentData()
        self.core.config.standings_export.fields = [
            self.ui.lw_fields.item(i).data(Qt.ItemDataRole.UserRole)
            for i
            in range(self.ui.lw_fields.count())
        ]
        self.core.config.standings_export.dir = self.ui.le_export_dir.text()
        self.core.config = self.core.config

        self.core.export_str(
            self.core.get_standings_str(),
            self.ui.le_export_dir.text(),
            StandingsExport.Target.FILE
        )

        self.close()

    @staticmethod
    def show_dialog(parent=None):
        dlg = ExportStandingsDialog(parent)
        dlg.show()
        result = dlg.exec()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--number_of_mock_players',
                        dest='number_of_mock_players', type=int, default=0)
    parser.add_argument('-s', '--pod_sizes', dest='pod_sizes', nargs='*', type=int, default=None,
                        help='Allowed od sizes by preference order (ex: "-s 4 3" will allow pods of size 4 and 3, preferring 4)')
    parser.add_argument('-b', '--allow_bye', dest='allow_bye',
                        action='store_true', default=False)
    parser.add_argument('-x', '--scoring', dest='scoring', nargs=3, type=int, default=None,
                        help='Change the scoring system. The first argument is the number of points for a win, the second is a draw, and the third is the number of points for a bye.')
    parser.add_argument('-S', '--snake', dest='snake', action='store_true', default=False)
    parser.add_argument('-r', '--rounds', dest='rounds', type=int, default=None,
                        help='Set the number of rounds for the tournament.')
    parser.add_argument('-o', '--open', dest='open', type=str, default=None,
                        help='Open a log file.')
    subparsers = parser.add_subparsers()
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)

    if args.open:
        core = TournamentAction.load(args.open)
    elif not (core:=TournamentAction.load()):
        core = Tournament()
        core.initialize_round()

    if args.pod_sizes:
        core.config.pod_sizes = args.pod_sizes
    if args.allow_bye:
        core.config.allow_bye = True
    if args.scoring:
        core.config.scoring(args.scoring)
    if args.number_of_mock_players:
        fkr = Faker()
        core.add_player([
            f"{i}:{fkr.name()}" for i in range(args.number_of_mock_players)
        ])
    if args.snake:
        core.config.snake_pods = True
    if args.rounds:
        core.config.n_rounds = args.rounds

    # for i in range(7):
    #   core.make_pods()
    #   core.random_results()

    window = MainWindow(core)
    window.show()

    app.exec()
    #for p in core.get_standings():
    #    print(p.unique_opponents)
    sys.exit(app.exit())
    # app.exec_()
    # sys.exit(app.exit())
