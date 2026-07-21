import matplotlib
from matplotlib.widgets import Button, TextBox
import matplotlib.pyplot as plt

def confirm(text):
    with matplotlib.rc_context({'toolbar': 'None'}):
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title("Notice")
        ax.axis('off')
        ax.text(0.5, 0.5, text, ha='center', va='center', wrap=True, fontsize=14)

        result = []

        # Add an "OK" button
        ax_ok = plt.axes([0.4, 0.1, 0.2, 0.1])
        btn_ok = Button(ax_ok, 'OK')

        def on_ok(event):
            result.append(True)
            plt.close(fig)

        btn_ok.on_clicked(on_ok)

        plt.show()
        return True if result else False


def call(question):
    with matplotlib.rc_context({'toolbar': 'None'}):
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title("Exit Application")
        ax.axis('off')
        ax.text(0.5, 0.6, question, ha='center', va='center', wrap=True, fontsize=14)

        result = []

        # Yes button
        ax_yes = plt.axes([0.25, 0.2, 0.2, 0.1])
        btn_yes = Button(ax_yes, 'Yes')
        def on_yes(event):
            result.append('yes')
            plt.close(fig)
        btn_yes.on_clicked(on_yes)

        # No button
        ax_no = plt.axes([0.55, 0.2, 0.2, 0.1])
        btn_no = Button(ax_no, 'No')
        def on_no(event):
            result.append('no')
            plt.close(fig)
        btn_no.on_clicked(on_no)

        plt.show()
        return result[0] if result else None


def ask(question):
    with matplotlib.rc_context({'toolbar': 'None'}):
        fig, ax = plt.subplots(figsize=(4, 2))
        fig.canvas.manager.set_window_title("Input")
        ax.axis('off')
        ax.text(0.5, 0.7, question, ha='center', va='center', wrap=True, fontsize=12)

        result = []

        # Text box
        ax_box = plt.axes([0.2, 0.4, 0.6, 0.2])
        text_box = TextBox(ax_box, "")

        # Submit button
        ax_submit = plt.axes([0.4, 0.1, 0.2, 0.15])
        btn_submit = Button(ax_submit, 'Submit')
        def on_submit(event):
            result.append(text_box.text)
            plt.close(fig)
        btn_submit.on_clicked(on_submit)

        plt.show()
        return result[0] if result else None
