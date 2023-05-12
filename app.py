import matplotlib
import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from streamlit_login_auth_ui.widgets import __login__
from pathlib import Path
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Create a login object
__login__obj = __login__(auth_token = "courier_auth_token",
                          company_name = "Shims",
                          width = 200, height = 250,
                          logout_button_name = 'Logout', hide_menu_bool = False,
                          hide_footer_bool = False,
                          lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()
username = __login__obj.get_username()

if LOGGED_IN == True:
    sns.set_style("darkgrid")
    matplotlib.rcParams['font.size'] = 20
    matplotlib.rcParams['figure.figsize'] = (6, 6)
    plt.savefig('msg.plots.svg', format='svg')
    matplotlib.rcParams['figure.facecolor'] = '#00000000'

    img = Image.open('pngegg.png')

    sentiments = SentimentIntensityAnalyzer()

    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #12961c;
        color:#ffffff;
    }
    div.stButton > button:hover {
        background-color: #21c315;
        color:##ff99ff;
        }
    </style>""", unsafe_allow_html=True)

    st.sidebar.title("Whatsapp Chat Analyzer")
    st.sidebar.image("https://i.pinimg.com/564x/d3/34/ee/d334ee01b200141cfa3bb9bc56186e10.jpg")
    uploaded_file = st.sidebar.file_uploader("Choose a file")

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf_8")
        df = preprocessor.preprocess(data)
        st.markdown("""
            <style>
            .big-font {
                font-size:65px !important;
            }
            </style>
            """, unsafe_allow_html=True)

        st.markdown('<p class="big-font">Whatsapp Chat Analyzer</p>', unsafe_allow_html=True)

        st.dataframe(df)

        # fetch unique users
        user_list = df['user'].unique().tolist()
        user_list.remove('ground_notification')
        user_list.sort()
        user_list.insert(0, "Overall")
        selected_user = st.sidebar.selectbox("Please Select Group Members", user_list)

        if st.sidebar.button("Show Analysis"):
            # Start Area
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
            st.title("Top Statistics")
            # Statistics
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.markdown("<h2 style='text-align: centre; color: #023020;'>Total Messages</h2>", unsafe_allow_html=True)

                st.title(num_messages)

            with col2:
                st.markdown("<h2 style='text-align: left; color: #023020;'>Total Words</h2>", unsafe_allow_html=True)
                st.title(words)

            with col3:
                st.markdown("<h2 style='text-align: left; color: #023020;'>Media Share</h2>", unsafe_allow_html=True)
                st.title(num_media_messages)

            with col4:
                st.markdown("<h2 style='text-align: left; color: #023020;'>Links Shared</h2>", unsafe_allow_html=True)
                st.title(num_links)

            # Monthly Timeline
            st.markdown("<h2 style='text-align: left; color: #023020;'>Monthly Timeline</h2>", unsafe_allow_html=True)

            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Daily Timeline
            st.markdown("<h2 style='text-align: left; color: #023020;'>Daily Timeline</h2>", unsafe_allow_html=True)
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Activity Map
            st.title("Activity Map")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<h2 style='text-align: left; color: #023020;'>Most Busy Day</h2>", unsafe_allow_html=True)
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='red')

                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.markdown("<h2 style='text-align: left; color: #023020;'>Most Busy Month</h2>", unsafe_allow_html=True)
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # finding the busiest user in the group(Group-level)
            if selected_user == 'Overall':
                st.markdown("<h2 style='text-align: left; color: #023020;'>Most Busy Users</h2>", unsafe_allow_html=True)
                x, new_df = helper.most_busy_users(df)
                fig, ax = plt.subplots()
                col1, col2 = st.columns(2)
                with col1:
                    ax.bar(x.index, x.values, color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)
            # WordCloud
            st.markdown("<h2 style='text-align: left; color: #023020;'>Word Cloud</h2>", unsafe_allow_html=True)
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            st.markdown("<h2 style='text-align: left; color: #023020;'>Weekly Activity Map</h2>", unsafe_allow_html=True)

            user_heatmap = helper.activity_heat_map(selected_user, df)
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

            # Emojis Analysis

            emoji_df = helper.emoji_helper(selected_user, df)
            st.markdown("<h2 style='text-align: left; color: #023020;'>Emoji Analysis</h2>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)

            # most common words
            st.markdown("<h2 style='text-align: left; color: #023020;'>Most Common Words</h2>", unsafe_allow_html=True)
            most_common_df = helper.most_common_words(selected_user, df)

            fig, ax = plt.subplots()

            ax.barh(most_common_df[0], most_common_df[1])

            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            st.markdown("<h2 style='text-align: left; color: #023020;'>Common Words In Numeric</h2>",
                        unsafe_allow_html=True)

            most_common_df = helper.most_common_words(selected_user, df)

            st.dataframe(most_common_df)
