import streamlit as st
import pandas as pd
import plotly.express as px


class MaternalHealthDashboard:
    def __init__(self):
        self.maternal_health_data = pd.read_csv(
            "data/Maternal Health Risk Data Set.csv")

        self.fetal_health_data = pd.read_csv(
            "data/fetal_health.csv")

    def create_dashboard(self):
        st.title("Maternal & Fetal Health Analytics Dashboard")
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        total = len(self.maternal_health_data)
        risk = self.maternal_health_data["RiskLevel"].str.lower()

        high = len(risk[risk == "high risk"])
        mid = len(risk[risk == "mid risk"])
        low = len(risk[risk == "low risk"])

        col1.metric("Total Patients", total)
        col2.metric("High Risk", high)
        col3.metric("Mid Risk", mid)
        col4.metric("Low Risk", low)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            self.create_pie_chart()

        with col2:
            self.create_bar_chart()

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            self.create_age_distribution()

        with col2:
            self.create_hr_distribution()

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            self.create_bs_distribution()

        with col2:
            self.create_bp_distribution()

        st.markdown("---")

        self.create_heatmap()

        st.markdown("---")

        self.create_dataset_preview()

    def create_pie_chart(self):
        fig = px.pie(
            self.maternal_health_data,
            names="RiskLevel",
            title="Maternal Risk Distribution",
            hole=0.45,)
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="content")

    def create_bar_chart(self):
        risk_count = self.maternal_health_data["RiskLevel"].value_counts()

        fig = px.bar(
            x=risk_count.index,
            y=risk_count.values,
            color=risk_count.index,
            title="Risk Level Distribution",
            labels={
                "x": "Risk Level",
                "y": "Number of Patients"
            }
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="content")


    def create_age_distribution(self):
        fig = px.histogram(
            self.maternal_health_data,
            x="Age",
            color="RiskLevel",
            nbins=20,
            title="Age Distribution")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="content")

    def create_bs_distribution(self):
        fig = px.box(
            self.maternal_health_data,
            x="RiskLevel",
            y="BS",
            color="RiskLevel",
            title="Blood Sugar vs Risk Level"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="content")


    def create_bp_distribution(self):
        fig = px.box(
            self.maternal_health_data,
            x="RiskLevel",
            y="DiastolicBP",
            color="RiskLevel",
            title="Diastolic Blood Pressure vs Risk Level"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="content")


    def create_hr_distribution(self):
        fig = px.histogram(
            self.maternal_health_data,
            x="HeartRate",
            color="RiskLevel",
            nbins=20,
            title="Heart Rate Distribution"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="content")

    def create_heatmap(self):
        corr = self.maternal_health_data.drop(columns=["RiskLevel"]).corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Feature Correlation Heatmap"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, width="content")

    def create_dataset_preview(self):
        st.subheader("Dataset Preview")
        st.dataframe(
            self.maternal_health_data.head(10),
            width="content")


if __name__ == "__main__":
    dashboard = MaternalHealthDashboard()
    dashboard.create_dashboard()