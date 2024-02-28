import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud


def run():
    st.header("Welcome to the Data Analysis page!")
    st.write("Here is a simple Exploratory Data Analysis study, followed by relevant visualizations.")

    # import visualisasi
    main_data = pd.read_csv('1429_1.csv')

    main_data2 = main_data.copy()

    # drop kolom tidak relevan dan memiliki data null semua
    main_data2 = main_data2.drop([
    'id',
    'name',
    'asins','keys','reviews.date',
    'reviews.dateAdded',
    'reviews.dateSeen',
    'reviews.didPurchase',
    'reviews.doRecommend',
    'reviews.id',
    'reviews.numHelpful','reviews.sourceURLs','reviews.userCity',
    'reviews.userProvince',
    'reviews.username'], axis = 1)

    # menambahkan kolom baru bernama sentiment.value
    main_data2['sentiment.value'] = np.where(main_data2['reviews.rating']<=2,'negative',
                                np.where((main_data2['reviews.rating']> 2) & (main_data2['reviews.rating'] <= 3),'neutral','positive'))
    
    st.write("\n")
    st.write("##### 1. The dataset used in this study")
    st.write("The purpose of this project is to classify reviews on Amazon products and delivery services by dividing them into three sentiments: negative, neutral, and positive.")
    st.dataframe(main_data2)

    # Group by 'sentiment.value' and count occurrences, renaming the count column
    pie_data = pd.DataFrame(main_data2['sentiment.value'].value_counts().reset_index())
    pie_data.rename(columns = {'index':'type_of_meal_plan',
                              'type_of_meal_plan':'count'}, inplace = True)

    def percentage_calculator(dataset,col_name):
        value_keeper = []
        for i in range(len(dataset)):
            calculate_value = dataset[col_name][i]/ sum(dataset[col_name])
            value_keeper.append(calculate_value)
        return value_keeper
    
        # definisikan label dan proporosinya
    label = ['negative','neutral','positive']
    proportions = percentage_calculator(pie_data,'count')


    # proses plot
    colors = ['lightpink','lightyellow','lightblue']
    plt.figure(figsize=(8, 8))
    fig, ax = plt.subplots(figsize=(8, 8))
    pie = ax.pie(proportions, labels=label, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.
    plt.legend()

    st.write("\n")
    st.write("##### 1. Check the proportions of review texts/comments based on sentiment values")
    st.write("Based on the pie chart, most customers are satisfied with our services and products due to high proportion of positive sentiments! (93%)")
    # display
    st.pyplot(fig)

    # 2. make new categories

    # siapkan nama baru
    new_name = ['Electronics', 'Kindle Products','Electronic Accessories']
    # initiate new column
    main_data2.insert(2,'new_categories','Others')
    main_data2.loc[(main_data2['categories'].str.contains('Tablet'))|(main_data2['categories'].str.contains('Electronics'))|(main_data2['categories'].str.contains('iPad'))|(main_data2['categories'] == 'Computers')|(main_data2['categories'] == 'Alexa')|(main_data2['categories'] == 'Echo'), 'new_categories'] = new_name[0]
    main_data2.loc[(main_data2['categories'].str.contains('Kindle'))|(main_data2['categories'].str.contains('Readers')), 'new_categories'] = new_name[1]
    main_data2.loc[(main_data2['categories'].str.contains('Accessories'))|(main_data2['categories'].str.contains('Adapters'))|(main_data2['categories'].str.contains('Cases'))|(main_data2['categories'].str.contains('Covers'))|(main_data2['categories'].str.contains('Storage'))|(main_data2['categories'].str.contains('Chargers')), 'new_categories'] = new_name[2]


    # membuat fungsi untuk menghapus nama brand supaya komentar saja yang terlihat
    def remove_words(semua_komentar, hapus_kata):
        for word in hapus_kata:
            semua_komentar = semua_komentar.replace(word, "")
        return semua_komentar
    
    def wordcloud_sentiment_based(dataset, col_name, sentiment_type):
        data_sentiment_type = dataset[dataset[col_name] == sentiment_type]
        grouping_new_categories = dataset['new_categories'].unique().tolist()
        
        for category in grouping_new_categories:
            wordcloud_data = data_sentiment_type[data_sentiment_type['new_categories'] == category]['reviews.text']
            if not wordcloud_data.empty:  # Check if there are words available
                string_wordcloud = ' '.join(str(item) for item in wordcloud_data.tolist())

                # Remove unnecessary words
                vocab_delete = ['Amazon', 'Kindle', 'tablet', 'Fire', 'app', 'Echo', 'Alexa', 'TV']
                string_wordcloud_new = remove_words(string_wordcloud, vocab_delete)

                # Generate wordcloud
                word_cloud_diagram = WordCloud(width=800, height=400, background_color='white', colormap='Pastel2').generate(string_wordcloud_new)

                # Plot wordcloud and save as image file
                plt.figure(figsize=(10, 5))
                plt.imshow(word_cloud_diagram, interpolation='bilinear')
                plt.axis('off')
                plt.savefig("wordcloud.png")  # Save the plot as an image file

                # Display the image using Streamlit
                st.write(f"Wordcloud for sentiment type '{sentiment_type}' for product category '{category.upper()}':")
                st.image("wordcloud.png", use_column_width=True)
            else:
                st.write(f"No words available for sentiment type '{sentiment_type}' and product category '{category.upper()}'.")


    
    # Create a select box with visualization options
    option = st.selectbox(
        'Select the sentiment value type for showing the wordclouds:',
        ('negative','neutral', 'positive')
    )

    # Display the selected visualization
    if option == 'negative':
        wordcloud_sentiment_based(main_data2,'sentiment.value','negative')
    elif option == 'neutral':
        wordcloud_sentiment_based(main_data2,'sentiment.value','neutral')
    else:
        wordcloud_sentiment_based(main_data2,'sentiment.value','positive')

if __name__ == '__main__':
    run()