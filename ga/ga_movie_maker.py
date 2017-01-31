from ga_algorithm import GA
from ga_utils import get_world, plot_route, find_fittest_island, migrate
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Barykina\\Downloads\\ffmpeg\\bin\\ffmpeg.exe'
import matplotlib.animation as manimation


def animate_island_model():
    world = get_world()
    ff_mpeg_writer = manimation.writers['ffmpeg']
    metadata = dict(title='Island-based Artificial Evolution', comment='Movie support!')
    writer = ff_mpeg_writer(fps=3, metadata=metadata)

    migration_interval = 1
    num_migrants = 1
    print 'Migration pattern: ' + str(migration_interval) + str(num_migrants)

    plt.ion()
    # init islands
    islands = []
    for i in range(0, 5):
        island = GA(world, 0.02, 5, True, 150, 'Island:' + str(i))
        islands.append(island)

    # prepare visualization
    fig = plt.figure()

    fig.subplots_adjust(left=0.2, wspace=0.2)
    ax1 = fig.add_subplot(231)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax2 = fig.add_subplot(232)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax3 = fig.add_subplot(233)
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax4 = fig.add_subplot(234)
    ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    ax5 = fig.add_subplot(235)
    ax5.xaxis.set_visible(False)
    ax5.yaxis.set_visible(False)

    with writer.saving(fig, "movies/islands2.mp4", 300):
        # run evolution
        for interval_num in range(1, 500 / migration_interval):
            # evaluate N populations on every island
            for island in islands:
                island.run(migration_interval)
                if interval_num % 18 == 0:
                    migrate(islands, num_migrants)
            # plot fittest for each island

            plot_route(ax1, islands[0].get_best())
            plot_route(ax2, islands[1].get_best())
            plot_route(ax3, islands[2].get_best())
            plot_route(ax4, islands[3].get_best())
            plot_route(ax5, islands[4].get_best())

            fittest = find_fittest_island(islands)
            print str(fittest.get_distance())
            plt.pause(0.1)
            writer.grab_frame()

if __name__ == "__main__":
    animate_island_model()
