import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def visualise(df, start_date, end_date, FILE, events_dict):
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Hive Analysis - {FILE}\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}', 
                fontsize=16)

    # Plot 1: Temperatures
    ax1.plot(df.index, df['t'], label='t')
    ax1.plot(df.index, df['t_i_1'], label='t_i_1')
    ax1.plot(df.index, df['t_i_2'], label='t_i_2')
    ax1.plot(df.index, df['t_i_3'], label='t_i_3')
    ax1.plot(df.index, df['t_i_4'], label='t_i_4')
    ax1.plot(df.index, df['t_i_5'], label='t_i_5')
    ax1.plot(df.index, df['t_o'], label='t_o')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Weight
    ax2.plot(df.index, df['weight_kg'], label='weight')
    ax2.set_ylabel('Weight (kg)')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Humidity
    ax3.plot(df.index, df['h'], label='humidity')
    ax3.set_ylabel('Humidity (%)')
    ax3.legend()
    ax3.grid(True)

    # # Add event markers to all plots
    # events_dict = {
    #     'Swarming': swarming_indexes,
    #     'Queencell': queencell_indexes,
    #     'Feeding': feeding_indexes,
    #     'Honey': honey_indexes,
    #     'Treatment': treatment_indexes,
    #     'Died': died_indexes
    # }

    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
    for ax in [ax1, ax2, ax3]:
        # Set x-axis limits
        ax.set_xlim(start_date, end_date)
        
        for (event_name, indexes), color in zip(events_dict.items(), colors):
            # Filter events to only show those within the date range
            filtered_indexes = [idx for idx in indexes if start_date <= idx <= end_date]
            for idx in filtered_indexes:
                ax.annotate(event_name, 
                        xy=(idx, ax.get_ylim()[1]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom',
                        color=color,
                        rotation=90)
                ax.axvline(x=idx, color=color, linestyle='--', alpha=0.3)

    # Format x-axis
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    plt.show()