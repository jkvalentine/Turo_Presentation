from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr as pearson


def clean_data(reservation_df, vehicle_df):
    '''
    Clean reservation and vehicle data frames for
    analysis and add a price difference feature

    Args:
        reservation_df: pandas data frame of reservation data for each vehicle
        vehicle_df: pandas data frame of vehicle attributes
    Returns:
        vehicle_df: updated pandas data frame oh vehicle attributes with
        reservation data
    '''
    # Get total number of reservations for each vehicle
    res = pd.get_dummies(reservation_df['reservation_type'], prefix='res_type')
    res['reservation'] = 1
    res_by_vehicle = res.groupby(level='vehicle_id').sum()

    vehicle_df['total_reservations'] = res_by_vehicle['reservation']
    res_types = ['res_type_1', 'res_type_2', 'res_type_3']
    vehicle_df[res_types] = res_by_vehicle[res_types]

    # Fill None values with 0's for vehicle_df that were never reserved
    vehicle_df.fillna(value=0, inplace=True)

    # Create a feature of the difference between actual and recommended price
    vehicle_df['price_difference'] = vehicle_df['actual_price'] - vehicle_df['recommended_price']

    return vehicle_df


def do_some_eda(vehicle_df):
    '''
    Perform some preliminary EDA to get an idea of the data

    Args:
        vehicle_df: pandas data frame of vehicle attributes
    Returns:
        None
    '''
    # What does our vehicle data look like in general?
    print vehicle_df.describe()

    # How do the continuous features correlate with reservations? 
    features_1 = ['actual_price',
                'recommended_price',
                'price_difference',
                'total_reservations']
    print vehicle_df[features_1].corr()

    # And how do the ordinal features correlate with reservations?
    features2 = ['num_images', 'description', 'total_reservations']
    print vehicle_df[features2].corr(method='spearman')


def plot_reservations_dist(vehicle_df):
    '''
    Create a plot of the distribution of total reservations

    Args:
        vehicle_df: pandas data frame of vehicle attributes
    Returns:
        None
    '''
    feature = vehicle_df['total_reservations']
    fig, ax = plt.subplots()
    sns.distplot(vehicle_df['total_reservations'], bins=range(25), hist=True)
    ax.set_ylabel('Probability')
    ax.axvline(x=feature.mean(), color='g', label='Mean')
    ax.axvline(x=feature.mean() + feature.std(), color='purple', label='STD', ls='dotted')
    ax.axvline(x=feature.mean() - feature.std(), color='purple', ls='dotted')
    ax.set_title('Total Number of Reservations Distribution')    
    ax.set_xlabel('Total Reservations')
    ax.legend()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('total_res_distr')


def make_box_plot(vehicle_df):
    '''
    Create a box plot of number of images vs. total reservations

    Args:
        vehicle_df: pandas data frame of vehicle attributes
    Returns:
        None
    '''
    fig, ax = plt.subplots()
    sns.boxplot(vehicle_df['num_images'], vehicle_df['total_reservations'])
    ax.set_ylabel('Total Reservations')
    ax.set_title('Total Reservations vs. Number of Images')
    ax.set_xlabel('Number of Images')
    ax.set_ylim(-1, 27)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('image_res_box_plot')


def make_reg_plot(vehicle_df):
    '''
    Create a plot relating price difference to total reservations
    showing the regression line

    Args:
        vehicle_df: pandas data frame of vehicle attributes
    Returns:
        None
    '''
    x = vehicle_df['price_difference']
    y = vehicle_df['total_reservations']
    stat = pearson(x, y)
    stats = "pearsonr= {:0.2f}; p={:0.2e}".format(stat[0], stat[1])
    fig, ax = plt.subplots()
    sns.regplot(x, y)
    ax.set_ylabel('Total Reservations')
    ax.set_xlabel('Price Difference')
    ax.set_title('Total Reservations vs. Price Difference')
    ax.annotate(stats, xy=(350, 320), xycoords='axes points')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('total_res_vs_price_diff')


def make_violin_plot(vehicle_df, feature, target, inner='box'):
    '''
    Create violin plots relating various features to different targets

    Args:
        vehicle_df: pandas data frame of vehicle attributes
        feature: (str) independent categorical variable of interest
        target: (str) dependent variable of interest
        inner: (str) 'box', 'quartiles', 'point', or 'stick'
        indicating data points interior to violin plot
    Returns:
        None
    '''
    fig, ax = plt.subplots()
    sns.violinplot(vehicle_df[feature],
                   vehicle_df[target],
                   inner=inner)

    if feature == 'technology' and target == 'total_reservations':
        tech_1 = vehicle_df[vehicle_df['technology'] == 1]['total_reservations']
        tech_0 = vehicle_df[vehicle_df['technology'] == 0]['total_reservations']
        plt.scatter(x=1, y=tech_1.mean(), color='white', label='Tech=1 Mean')
        plt.scatter(x=0, y=tech_0.mean(), color='black', label='Tech=0 Mean')
        ax.set_ylabel('Total Reservations')
        ax.set_xlabel('Technology')
        ax.set_title('KDE of Total Reservations vs. Technology \n With Quartiles Marked')
        ax.legend(loc=9)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)
        plt.savefig('total_res_vs_technology')

    elif feature == 'street_parked' and target == 'total_reservations':
        ax.set_ylabel('Total Reservations')
        ax.set_xlabel('Street Parked')
        ax.set_xticklabels(['No', 'Yes'])
        ax.set_title('KDE of Total Reservations vs. Street Parked')
        plt.savefig('total_res_vs_street_parked')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)

    elif feature == 'technology':
        ax.set_xlabel('Technology')
        if target == 'res_type_1':
            ax.set_ylabel('Number of Hourly Rentals')
            ax.set_title('KDE of Hourly Rentals vs. Technology')
            plt.savefig('hourly_res_vs_technology')
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)

        elif target == 'res_type_2':
            ax.set_ylabel('Number of Daily Rentals')
            ax.set_title('KDE of Daily Rentals vs. Technology')
            plt.savefig('daily_res_vs_technology')
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)

        elif target == 'res_type_3':
            ax.set_ylabel('Number of Weekly Rentals')
            ax.set_title('KDE of Weekly Rentals vs. Technology')
            plt.savefig('weekly_res_vs_technology')
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.5)


if __name__ == '__main__':
    # Read data into pandas data frames
    vehicle_df = pd.read_csv('vehicles_(6).csv', index_col='vehicle_id')
    reservation_df = pd.read_csv('reservations_(5).csv', index_col='vehicle_id')

    # Clean the data and add price difference features
    vehicle_df = clean_data(reservation_df, vehicle_df)

    # Do some EDA to get an idea of what the data looks like and what
    # features correlate with others
    do_some_eda(vehicle_df)

    # Create some plots to see how the data relates visually
    # First, how the reservations are distributed
    plot_reservations_dist(vehicle_df)

    # Make a box-plot to look at the distribution of
    # number of images
    make_box_plot(vehicle_df)

    # Make a scatter plot relating difference between actual and
    # recommended price to number of reservations, showing regression
    # line and each feature's distribution
    make_joint_plot(vehicle_df)

    # Make a violin plot to relate technology type to total reservations
    make_violin_plot(vehicle_df, 'technology', 'total_reservations', inner='quartiles')

    # Make a violin plot to relate street parking to total reservations
    make_violin_plot(vehicle_df, 'street_parked', 'total_reservations', inner='box')    

    # Make a series of violin plots looking at how technology affects
    # each type of reservation
    make_violin_plot(vehicle_df, 'technology', 'res_type_1', inner='box')
    make_violin_plot(vehicle_df, 'technology', 'res_type_2', inner='box')
    make_violin_plot(vehicle_df, 'technology', 'res_type_3', inner='box')
