#
import os
import sys
import json
#
from matplotlib import pyplot as plt
#



if __name__ == "__main__":

    #
    if len(sys.argv) != 3:
        #
        raise UserWarning(f"Bad arguments : {sys.argv}")

    #
    json_file: str = sys.argv[1]
    where_to_save_plot: str = sys.argv[2]

    #
    if not os.path.exists(json_file):
        #
        raise FileNotFoundError(f"{json_file}")

    #
    with open(json_file, "r", encoding="utf-8") as f:
        #
        data: dict[str, list[float]] = json.load(f)

    #
    for fig_name in data.keys():
        #
        plt.plot( data[fig_name], label=fig_name )

    #
    plt.legend(loc="upper right")
    #
    plt.savefig( where_to_save_plot )
    #
    plt.close()
