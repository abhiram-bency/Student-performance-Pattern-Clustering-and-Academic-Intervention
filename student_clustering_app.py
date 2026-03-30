import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Performance Clustering",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
    }
    .cluster-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .intervention-box {
        border-left: 4px solid #ff4b4b;
        padding-left: 1rem;
        margin: 1rem 0;
        background-color: #fff5f5;
        border-radius: 5px;
    }
    .success-box {
        border-left: 4px solid #00c853;
        padding-left: 1rem;
        margin: 1rem 0;
        background-color: #f1f8f4;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Column mapping
COLUMN_MAPPING = {
    'cgpa': 'What is your current CGPA?',
    'attendance': 'Average attendance on class',
    'previous_sgpa': 'What was your previous SGPA?',
    'credits_completed': 'How many Credit did you have completed?',
    'study_hours': 'How many hour do you study daily?',
    'social_hours': 'How many hour do you spent daily in social media?',
    'skill_hours': 'How many hour do you spent daily on your skill development?',
    'probation': 'Did you ever fall in probation?',
    'suspension': 'Did you ever got suspension?',
    'scholarship': 'Do you have meritorious scholarship ?',
    'consultancy': 'Do you attend in teacher consultancy for any kind of academical problems?',
    'current_semester': 'Current Semester',
    'family_income': 'What is your monthly family income?',
    'age': 'Age',
    'english_proficiency': 'Status of your English language proficiency',
    'co_curricular': 'Are you engaged with any co-curriculum activities?'
}

@st.cache_data
def load_and_preprocess_data(df):
    """Comprehensive data preprocessing"""
    df_clean = df.copy()
    
    # Handle missing values
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
    
    # Feature engineering
    study_col = COLUMN_MAPPING.get('study_hours')
    sessions_col = 'How many times do you seat for study in a day?'
    
    if study_col in df_clean.columns and sessions_col in df_clean.columns:
        df_clean['Study_Efficiency'] = df_clean[study_col] / (df_clean[sessions_col] + 1)
    else:
        df_clean['Study_Efficiency'] = 1.0
    
    # Academic risk score
    risk_factors = []
    probation_col = COLUMN_MAPPING.get('probation')
    if probation_col in df_clean.columns:
        df_clean[probation_col] = (df_clean[probation_col] > 0).astype(int)
        risk_factors.append(df_clean[probation_col] * 2)
    
    suspension_col = COLUMN_MAPPING.get('suspension')
    if suspension_col in df_clean.columns:
        df_clean[suspension_col] = (df_clean[suspension_col] > 0).astype(int)
        risk_factors.append(df_clean[suspension_col] * 3)
    
    attendance_col = COLUMN_MAPPING.get('attendance')
    if attendance_col in df_clean.columns:
        attendance = df_clean[attendance_col].clip(0, 100)
        attendance_risk = (100 - attendance) / 25
        risk_factors.append(attendance_risk)
    
    if risk_factors:
        df_clean['Academic_Risk_Score'] = sum(risk_factors)
    else:
        df_clean['Academic_Risk_Score'] = 0
    
    # Time management score
    study_col = COLUMN_MAPPING.get('study_hours')
    social_col = COLUMN_MAPPING.get('social_hours')
    
    if study_col in df_clean.columns and social_col in df_clean.columns:
        df_clean['Time_Management_Score'] = df_clean[study_col] - df_clean[social_col]
    else:
        df_clean['Time_Management_Score'] = 0
    
    # Select features for clustering
    cluster_features = ['Academic_Risk_Score', 'Time_Management_Score', 'Study_Efficiency']
    
    if COLUMN_MAPPING['cgpa'] in df_clean.columns:
        cluster_features.append(COLUMN_MAPPING['cgpa'])
    
    key_features = [
        COLUMN_MAPPING.get('attendance'),
        COLUMN_MAPPING.get('study_hours'),
        COLUMN_MAPPING.get('social_hours'),
        COLUMN_MAPPING.get('scholarship'),
        COLUMN_MAPPING.get('probation'),
        COLUMN_MAPPING.get('credits_completed'),
        COLUMN_MAPPING.get('consultancy'),
        COLUMN_MAPPING.get('current_semester'),
        COLUMN_MAPPING.get('family_income')
    ]
    
    for feature in key_features:
        if feature and feature in df_clean.columns and feature not in cluster_features:
            cluster_features.append(feature)
    
    df_cluster = df_clean[cluster_features].copy()
    
    # Ensure numeric
    for col in df_cluster.columns:
        df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce')
    df_cluster = df_cluster.fillna(df_cluster.median())
    
    # Handle outliers
    for col in df_cluster.columns:
        Q1 = df_cluster[col].quantile(0.25)
        Q3 = df_cluster[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_cluster[col] = np.where(df_cluster[col] < lower_bound, lower_bound,
                                      np.where(df_cluster[col] > upper_bound, upper_bound, df_cluster[col]))
    
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_cluster)
    scaled_df = pd.DataFrame(scaled_data, columns=df_cluster.columns)
    
    return scaled_df, df_clean, label_encoders, scaler, cluster_features

@st.cache_data
def apply_dimensionality_reduction(data, method='pca', n_components=3):
    """Apply PCA or t-SNE"""
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        reduced_data = reducer.fit_transform(data)
        columns = [f'PC_{i+1}' for i in range(n_components)]
        variance_ratio = reducer.explained_variance_ratio_
    else:
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        reduced_data = reducer.fit_transform(data)
        columns = [f'tSNE_{i+1}' for i in range(n_components)]
        variance_ratio = None
    
    reduced_df = pd.DataFrame(reduced_data, columns=columns)
    return reduced_df, reducer, variance_ratio

@st.cache_data
def find_optimal_clusters(data, max_clusters=10):
    """Determine optimal number of clusters"""
    metrics = {
        'n_clusters': [],
        'wcss': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(data)
        
        metrics['n_clusters'].append(n_clusters)
        metrics['wcss'].append(kmeans.inertia_)
        
        if len(set(labels)) > 1:
            metrics['silhouette'].append(silhouette_score(data, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(data, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(data, labels))
        else:
            metrics['silhouette'].append(0)
            metrics['davies_bouldin'].append(float('inf'))
            metrics['calinski_harabasz'].append(0)
    
    optimal_k = np.argmax(metrics['silhouette']) + 2
    return metrics, optimal_k

@st.cache_data
def apply_clustering(data, n_clusters=4, algorithm='KMeans'):
    """Apply selected clustering algorithm"""
    if algorithm == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
    elif algorithm == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:  # DBSCAN
        model = DBSCAN(eps=0.5, min_samples=5)
    
    labels = model.fit_predict(data)
    
    if len(set(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
    else:
        silhouette = davies_bouldin = calinski_harabasz = 0
    
    return labels, model, {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz
    }

def get_cluster_profiles(df_original, cluster_labels):
    """Generate cluster profiles"""
    df_analysis = df_original.copy()
    df_analysis['Cluster'] = cluster_labels
    
    clusters = sorted(df_analysis['Cluster'].unique())
    profiles = {}
    
    for cluster in clusters:
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        profile = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_analysis) * 100
        }
        
        # Calculate metrics
        metrics = [
            ('CGPA', COLUMN_MAPPING.get('cgpa')),
            ('Attendance', COLUMN_MAPPING.get('attendance')),
            ('Study Hours', COLUMN_MAPPING.get('study_hours')),
            ('Social Media Hours', COLUMN_MAPPING.get('social_hours')),
            ('Academic Risk Score', 'Academic_Risk_Score'),
            ('Time Management Score', 'Time_Management_Score'),
            ('Study Efficiency', 'Study_Efficiency')
        ]
        
        for metric_name, col_name in metrics:
            if col_name and col_name in cluster_data.columns:
                values = pd.to_numeric(cluster_data[col_name], errors='coerce').dropna()
                if len(values) > 0:
                    profile[metric_name] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max()
                    }
        
        # Categorical metrics
        cat_metrics = [
            ('Probation Rate', COLUMN_MAPPING.get('probation')),
            ('Scholarship Rate', COLUMN_MAPPING.get('scholarship')),
            ('Consultancy Rate', COLUMN_MAPPING.get('consultancy'))
        ]
        
        for metric_name, col_name in cat_metrics:
            if col_name and col_name in cluster_data.columns:
                values = cluster_data[col_name]
                if values.dtype == 'object':
                    values = values.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0})
                values = pd.to_numeric(values, errors='coerce').dropna()
                if len(values) > 0:
                    profile[metric_name] = values.mean() * 100
        
        profiles[cluster] = profile
    
    return profiles, df_analysis

def get_cluster_characterization(profile):
    """Characterize a cluster based on its profile"""
    characteristics = []
    
    cgpa = profile.get('CGPA', {}).get('mean', 0)
    if cgpa >= 3.7:
        characteristics.append("🌟 Excellent Performance (CGPA ≥ 3.7)")
        characteristics.append("🎓 Honors/Dean's List Candidates")
    elif cgpa >= 3.3:
        characteristics.append("📈 Very Good Performance (CGPA 3.3-3.69)")
        characteristics.append("💪 Strong Academic Standing")
    elif cgpa >= 3.0:
        characteristics.append("✅ Good Performance (CGPA 3.0-3.29)")
        characteristics.append("📚 Meeting Expectations")
    elif cgpa >= 2.7:
        characteristics.append("⚠️ Needs Improvement (CGPA 2.7-2.99)")
        characteristics.append("🔍 Monitor Closely")
    elif cgpa >= 2.0:
        characteristics.append("🚨 At-Risk (CGPA 2.0-2.69)")
        characteristics.append("🆘 Requires Intervention")
    else:
        characteristics.append("🔴 Critical (CGPA < 2.0)")
        characteristics.append("🏥 Academic Probation Likely")
    
    attendance = profile.get('Attendance', {}).get('mean', 0)
    if attendance >= 90:
        characteristics.append("✅ Excellent Attendance (≥90%)")
    elif attendance >= 80:
        characteristics.append("📊 Good Attendance (80-89%)")
    elif attendance >= 70:
        characteristics.append("⚠️ Fair Attendance (70-79%)")
    elif attendance >= 60:
        characteristics.append("🔔 Needs Improvement (60-69%)")
    else:
        characteristics.append("🚨 Poor Attendance (<60%)")
    
    risk_score = profile.get('Academic Risk Score', {}).get('mean', 0)
    if risk_score > 5:
        characteristics.append("🔴 High Academic Risk")
    elif risk_score > 2:
        characteristics.append("🟡 Moderate Risk")
    else:
        characteristics.append("🟢 Low Risk")
    
    return characteristics

def get_intervention_strategies(profile, cluster_id):
    """Generate intervention strategies for a cluster"""
    cgpa = profile.get('CGPA', {}).get('mean', 0)
    attendance = profile.get('Attendance', {}).get('mean', 0)
    study_hours = profile.get('Study Hours', {}).get('mean', 0)
    social_hours = profile.get('Social Media Hours', {}).get('mean', 0)
    risk_score = profile.get('Academic Risk Score', {}).get('mean', 0)
    
    strategies = {
        'Academic Performance': [],
        'Attendance Management': [],
        'Study Habits': [],
        'Risk Mitigation': [],
        'Implementation': {}
    }
    
    # Academic Performance
    if cgpa < 2.0:
        strategies['Academic Performance'] = [
            "🚨 CRITICAL: Mandatory academic probation meeting",
            "📚 Required: 6+ hours tutoring per week",
            "👥 Assign: Academic mentor & peer tutor",
            "📋 Implement: Daily progress tracking",
            "🏫 Enforce: 100% class attendance requirement",
            "📞 Required: Weekly parent/guardian updates"
        ]
    elif cgpa < 2.5:
        strategies['Academic Performance'] = [
            "⚠️ HIGH PRIORITY: Academic counseling (bi-weekly)",
            "📚 Required: 4+ hours tutoring per week",
            "👥 Assign: Peer mentor",
            "📋 Implement: Weekly progress review",
            "🏫 Enforce: 90% minimum attendance"
        ]
    elif cgpa < 3.0:
        strategies['Academic Performance'] = [
            "📈 MODERATE: Study skills workshop (monthly)",
            "📚 Optional: 2+ hours tutoring per week",
            "⏰ Training: Time management workshop",
            "🎯 Set: Specific GPA improvement goals",
            "🔍 Conduct: Mid-term academic review"
        ]
    elif cgpa < 3.5:
        strategies['Academic Performance'] = [
            "💪 MAINTENANCE: Advanced study techniques",
            "🎓 Explore: Research opportunities",
            "📊 Provide: Career planning sessions",
            "🌟 Encourage: Leadership development",
            "🎯 Aim: Dean's List qualification"
        ]
    else:
        strategies['Academic Performance'] = [
            "🏆 EXCELLENCE: Honors program invitation",
            "🔬 Offer: Undergraduate research positions",
            "👨‍🏫 Provide: Teaching assistant opportunities",
            "💼 Facilitate: Competitive internship placements",
            "🎓 Prepare: Graduate school applications"
        ]
    
    # Attendance Management
    if attendance < 70:
        strategies['Attendance Management'] = [
            "✅ Implement: Daily attendance tracking",
            "📱 Set up: Automated absence notifications",
            "📞 Require: Meeting after 3+ absences",
            "👨‍👩‍👧 Involve: Parent/guardian for <60% attendance"
        ]
    elif attendance < 80:
        strategies['Attendance Management'] = [
            "📊 Monitor: Attendance patterns weekly",
            "💬 Provide: Early warning for low attendance",
            "🎯 Set: Attendance improvement goals"
        ]
    elif attendance < 90:
        strategies['Attendance Management'] = [
            "👍 Encourage: Perfect attendance rewards",
            "🔔 Provide: Gentle reminders"
        ]
    else:
        strategies['Attendance Management'] = [
            "🎉 Recognize: Excellent attendance record",
            "📜 Provide: Attendance certificate"
        ]
    
    # Study Habits
    if study_hours < 2:
        strategies['Study Habits'] = [
            "⏳ Enforce: Minimum 2 hours daily study",
            "📅 Create: Structured study schedule",
            "🏫 Provide: Designated study space access",
            "👥 Pair: With study buddy"
        ]
    elif social_hours > study_hours:
        strategies['Study Habits'] = [
            "📱 Limit: Social media during study hours",
            "⏰ Teach: Pomodoro technique",
            "🎯 Set: Study vs social media ratio goals"
        ]
    elif study_hours > 4:
        strategies['Study Habits'] = [
            "⚖️ Balance: Ensure study-life balance",
            "🧘 Encourage: Stress management techniques",
            "🏃 Promote: Physical activity breaks"
        ]
    else:
        strategies['Study Habits'] = [
            "👍 Maintain: Current study patterns",
            "💡 Share: Best practices with peers"
        ]
    
    # Risk Mitigation
    if risk_score > 7:
        strategies['Risk Mitigation'] = [
            "🚑 URGENT: Intensive academic support program",
            "🏥 Refer: To counseling services",
            "📋 Create: Individualized recovery plan",
            "👨‍👩‍👧‍👦 Involve: Full support team (academic, counseling, family)"
        ]
    elif risk_score > 4:
        strategies['Risk Mitigation'] = [
            "🔴 HIGH: Weekly check-ins with advisor",
            "📚 Provide: Additional academic resources",
            "🎯 Develop: Risk reduction plan"
        ]
    elif risk_score > 2:
        strategies['Risk Mitigation'] = [
            "🟡 MODERATE: Monthly progress reviews",
            "💡 Offer: Proactive academic support"
        ]
    else:
        strategies['Risk Mitigation'] = [
            "🟢 LOW: Maintain current support level",
            "👍 Encourage: Peer mentoring of at-risk students"
        ]
    
    # Implementation details
    strategies['Implementation'] = {
        'Review Schedule': 'Weekly' if risk_score > 4 or cgpa < 2.5 else 'Bi-weekly' if risk_score > 2 or cgpa < 3.0 else 'Monthly',
        'Contact Point': 'Academic Advisor + ' + ('Counselor' if risk_score > 4 else 'Department Mentor'),
        'Success Metrics': 'CGPA improvement + attendance' if cgpa < 3.0 else 'Research/output quality'
    }
    
    return strategies

# Main app
def main():
    st.markdown('<div class="main-header">🎓 Student Performance Pattern Clustering<br>for Academic Intervention</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Configuration")
        
        uploaded_file = st.file_uploader("Upload Student Data (CSV)", type=['csv'])
        
        st.markdown("---")
        st.subheader("Clustering Settings")
        
        reduction_method = st.selectbox("Dimensionality Reduction", ["PCA", "t-SNE"])
        n_components = st.slider("Number of Components", 2, 3, 3)
        
        algorithm = st.selectbox("Clustering Algorithm", ["KMeans", "Agglomerative", "DBSCAN"])
        
        if algorithm != 'DBSCAN':
            n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        else:
            st.info("DBSCAN automatically determines clusters")
            n_clusters = None
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This application performs clustering analysis on student performance data 
        to identify distinct groups and provide targeted academic intervention strategies.
        
        **Features:**
        - Interactive visualizations
        - Cluster profiling
        - Personalized interventions
        - Exportable results
        """)
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            with st.spinner("Processing data..."):
                scaled_df, df_processed, encoders, scaler, features = load_and_preprocess_data(df)
            
            st.success(f"✅ Data loaded: {len(df)} students with {len(df.columns)} features")
            
            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Data Overview", 
                "🔍 Optimal Clusters", 
                "🎯 Clustering Results", 
                "📈 Cluster Analysis",
                "💡 Intervention Strategies"
            ])
            
            # Tab 1: Data Overview
            with tab1:
                st.header("Dataset Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Students", len(df))
                with col2:
                    st.metric("Total Features", len(df.columns))
                with col3:
                    st.metric("Features Selected", len(features))
                with col4:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                st.subheader("Sample Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.subheader("Feature Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    cgpa_col = COLUMN_MAPPING.get('cgpa')
                    if cgpa_col in df.columns:
                        fig = px.histogram(df, x=cgpa_col, nbins=30, 
                                         title="CGPA Distribution",
                                         labels={cgpa_col: 'CGPA'},
                                         color_discrete_sequence=['#1f77b4'])
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    attendance_col = COLUMN_MAPPING.get('attendance')
                    if attendance_col in df.columns:
                        fig = px.histogram(df, x=attendance_col, nbins=30,
                                         title="Attendance Distribution",
                                         labels={attendance_col: 'Attendance (%)'},
                                         color_discrete_sequence=['#ff7f0e'])
                        st.plotly_chart(fig, use_container_width=True)
            
            # Tab 2: Optimal Clusters
            with tab2:
                st.header("Finding Optimal Number of Clusters")
                
                with st.spinner("Calculating optimal clusters..."):
                    reduced_df, reducer, variance = apply_dimensionality_reduction(
                        scaled_df, reduction_method.lower(), n_components
                    )
                    metrics, optimal_k = find_optimal_clusters(reduced_df, max_clusters=10)
                
                st.success(f"📊 Suggested optimal clusters: **{optimal_k}**")
                
                if variance is not None:
                    st.info(f"📈 PCA Explained Variance: {sum(variance):.2%}")
                
                # Plot metrics
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Elbow Method (WCSS)", "Silhouette Score", 
                                   "Davies-Bouldin Index", "Calinski-Harabasz Score")
                )
                
                fig.add_trace(
                    go.Scatter(x=metrics['n_clusters'], y=metrics['wcss'], 
                              mode='lines+markers', name='WCSS',
                              line=dict(color='blue', width=2)),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=metrics['n_clusters'], y=metrics['silhouette'],
                              mode='lines+markers', name='Silhouette',
                              line=dict(color='red', width=2)),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(x=metrics['n_clusters'], y=metrics['davies_bouldin'],
                              mode='lines+markers', name='Davies-Bouldin',
                              line=dict(color='green', width=2)),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=metrics['n_clusters'], y=metrics['calinski_harabasz'],
                              mode='lines+markers', name='Calinski-Harabasz',
                              line=dict(color='purple', width=2)),
                    row=2, col=2
                )
                
                fig.update_xaxes(title_text="Number of Clusters", row=2, col=1)
                fig.update_xaxes(title_text="Number of Clusters", row=2, col=2)
                
                fig.update_layout(height=700, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Interpretation:**
                - **Elbow Method**: Look for the "elbow" where decrease slows
                - **Silhouette Score**: Higher is better (closer to 1)
                - **Davies-Bouldin Index**: Lower is better
                - **Calinski-Harabasz**: Higher is better
                """)
            
            # Tab 3: Clustering Results
            with tab3:
                st.header("Clustering Results")
                
                if algorithm != 'DBSCAN':
                    actual_n_clusters = n_clusters
                else:
                    actual_n_clusters = None
                
                with st.spinner(f"Applying {algorithm} clustering..."):
                    labels, model, cluster_metrics = apply_clustering(
                        reduced_df, actual_n_clusters, algorithm
                    )
                
                n_clusters_found = len(set(labels))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Clusters Found", n_clusters_found)
                with col2:
                    st.metric("Silhouette Score", f"{cluster_metrics['silhouette']:.3f}")
                with col3:
                    st.metric("Davies-Bouldin Index", f"{cluster_metrics['davies_bouldin']:.3f}")
                
                # Cluster distribution
                st.subheader("Cluster Distribution")
                cluster_counts = pd.Series(labels).value_counts().sort_index()
                
                fig = go.Figure(data=[
                    go.Bar(x=[f'Cluster {i}' for i in cluster_counts.index],
                          y=cluster_counts.values,
                          text=[f'{v}<br>({v/len(labels)*100:.1f}%)' for v in cluster_counts.values],
                          textposition='auto',
                          marker_color=px.colors.qualitative.Set3[:len(cluster_counts)])
                ])
                fig.update_layout(
                    title="Number of Students per Cluster",
                    xaxis_title="Cluster",
                    yaxis_title="Number of Students",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 3D Visualization
                if n_components == 3:
                    st.subheader("3D Cluster Visualization")
                    
                    plot_df = reduced_df.copy()
                    plot_df['Cluster'] = [f'Cluster {l}' for l in labels]
                    
                    fig = px.scatter_3d(
                        plot_df,
                        x=plot_df.columns[0],
                        y=plot_df.columns[1],
                        z=plot_df.columns[2],
                        color='Cluster',
                        title=f"{algorithm} Clustering - 3D View",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        height=700
                    )
                    fig.update_traces(marker=dict(size=5))
                    st.plotly_chart(fig, use_container_width=True)
                
                # 2D Visualization
                st.subheader("2D Cluster Visualization")
                
                plot_df_2d = reduced_df.iloc[:, :2].copy()
                plot_df_2d['Cluster'] = [f'Cluster {l}' for l in labels]
                
                fig = px.scatter(
                    plot_df_2d,
                    x=plot_df_2d.columns[0],
                    y=plot_df_2d.columns[1],
                    color='Cluster',
                    title=f"{algorithm} Clustering - 2D View",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    height=600
                )
                fig.update_traces(marker=dict(size=8))
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 4: Cluster Analysis
            with tab4:
                st.header("Detailed Cluster Analysis")
                
                profiles, df_with_clusters = get_cluster_profiles(df_processed, labels)
                
                for cluster_id in sorted(profiles.keys()):
                    profile = profiles[cluster_id]
                    
                    st.markdown(f'<div class="cluster-header">Cluster {cluster_id}: {profile["size"]} students ({profile["percentage"]:.1f}%)</div>', 
                              unsafe_allow_html=True)
                    
                    # Key metrics
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    
                    with metrics_col1:
                        cgpa = profile.get('CGPA', {}).get('mean', 0)
                        st.metric("Average CGPA", f"{cgpa:.2f}")
                    
                    with metrics_col2:
                        attendance = profile.get('Attendance', {}).get('mean', 0)
                        st.metric("Average Attendance", f"{attendance:.1f}%")
                    
                    with metrics_col3:
                        study = profile.get('Study Hours', {}).get('mean', 0)
                        st.metric("Study Hours/Day", f"{study:.1f}")
                    
                    with metrics_col4:
                        risk = profile.get('Academic Risk Score', {}).get('mean', 0)
                        st.metric("Risk Score", f"{risk:.1f}/10")
                    
                    # Characteristics
                    st.subheader("Cluster Characteristics")
                    characteristics = get_cluster_characterization(profile)
                    for char in characteristics:
                        st.markdown(f"- {char}")
                    
                    # Detailed metrics
                    with st.expander("View Detailed Metrics"):
                        metric_names = ['CGPA', 'Attendance', 'Study Hours', 'Social Media Hours',
                                       'Academic Risk Score', 'Time Management Score', 'Study Efficiency']
                        
                        for metric in metric_names:
                            if metric in profile and isinstance(profile[metric], dict):
                                st.markdown(f"**{metric}:**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.write(f"Mean: {profile[metric]['mean']:.2f}")
                                with col2:
                                    st.write(f"Std: {profile[metric]['std']:.2f}")
                                with col3:
                                    st.write(f"Min: {profile[metric]['min']:.2f}")
                                with col4:
                                    st.write(f"Max: {profile[metric]['max']:.2f}")
                    
                    st.markdown("---")
                
                # Comparison charts
                st.subheader("Cluster Comparison")
                
                comparison_metrics = ['CGPA', 'Attendance', 'Study Hours', 'Academic Risk Score']
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=comparison_metrics
                )
                
                row_col = [(1,1), (1,2), (2,1), (2,2)]
                
                for idx, metric in enumerate(comparison_metrics):
                    values = []
                    cluster_ids = []
                    
                    for cluster_id in sorted(profiles.keys()):
                        if metric in profiles[cluster_id] and isinstance(profiles[cluster_id][metric], dict):
                            values.append(profiles[cluster_id][metric]['mean'])
                            cluster_ids.append(f'Cluster {cluster_id}')
                    
                    if values:
                        row, col = row_col[idx]
                        fig.add_trace(
                            go.Bar(x=cluster_ids, y=values, name=metric,
                                  marker_color=px.colors.qualitative.Set3[idx]),
                            row=row, col=col
                        )
                
                fig.update_layout(height=700, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 5: Intervention Strategies
            with tab5:
                st.header("Targeted Intervention Strategies")
                
                for cluster_id in sorted(profiles.keys()):
                    profile = profiles[cluster_id]
                    
                    st.markdown(f'<div class="cluster-header">Intervention Plan for Cluster {cluster_id}</div>', 
                              unsafe_allow_html=True)
                    
                    st.markdown(f"**Students:** {profile['size']} ({profile['percentage']:.1f}% of total)")
                    
                    strategies = get_intervention_strategies(profile, cluster_id)
                    
                    # Display strategies
                    for category, items in strategies.items():
                        if category != 'Implementation':
                            st.subheader(f"📌 {category}")
                            
                            if isinstance(items, list):
                                for item in items:
                                    # Determine box style based on urgency
                                    if '🚨' in item or '🔴' in item or 'CRITICAL' in item or 'URGENT' in item:
                                        st.markdown(f'<div class="intervention-box">• {item}</div>', 
                                                  unsafe_allow_html=True)
                                    elif '🏆' in item or '🌟' in item or 'EXCELLENCE' in item:
                                        st.markdown(f'<div class="success-box">• {item}</div>', 
                                                  unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"- {item}")
                    
                    # Implementation details
                    st.subheader("📋 Implementation Details")
                    impl = strategies['Implementation']
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.info(f"**Review Schedule:**\n\n{impl['Review Schedule']}")
                    with col2:
                        st.info(f"**Contact Point:**\n\n{impl['Contact Point']}")
                    with col3:
                        st.info(f"**Success Metrics:**\n\n{impl['Success Metrics']}")
                    
                    st.markdown("---")
                
                # Overall recommendations
                st.markdown('<div class="cluster-header">📊 Overall Recommendations</div>', 
                          unsafe_allow_html=True)
                
                at_risk_clusters = [c for c, p in profiles.items() 
                                   if p.get('CGPA', {}).get('mean', 4) < 2.5 or 
                                   p.get('Academic Risk Score', {}).get('mean', 0) > 5]
                
                if at_risk_clusters:
                    st.error(f"""
                    **🚨 PRIORITY ACTION NEEDED:**
                    - {len(at_risk_clusters)} cluster(s) require immediate attention
                    - Focus resources on Clusters: {', '.join(map(str, at_risk_clusters))}
                    - Consider implementing early warning system
                    - Schedule emergency intervention meetings
                    """)
                
                successful_clusters = [c for c, p in profiles.items() 
                                     if p.get('CGPA', {}).get('mean', 0) > 3.5 and 
                                     p.get('Academic Risk Score', {}).get('mean', 10) < 3]
                
                if successful_clusters:
                    st.success(f"""
                    **✅ SUCCESS MODELS:**
                    - Clusters {', '.join(map(str, successful_clusters))} show excellent performance
                    - Study their habits and share best practices
                    - Consider peer mentoring program with these students
                    - Use as models for intervention success stories
                    """)
                
                # Download results
                st.subheader("📥 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = df_with_clusters.to_csv(index=False)
                    st.download_button(
                        label="Download Clustered Data (CSV)",
                        data=csv,
                        file_name="student_clusters.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    profiles_df = pd.DataFrame([
                        {
                            'Cluster': k,
                            'Size': v['size'],
                            'Percentage': v['percentage'],
                            'Avg_CGPA': v.get('CGPA', {}).get('mean', 0),
                            'Avg_Attendance': v.get('Attendance', {}).get('mean', 0),
                            'Risk_Score': v.get('Academic Risk Score', {}).get('mean', 0)
                        }
                        for k, v in profiles.items()
                    ])
                    
                    csv_profiles = profiles_df.to_csv(index=False)
                    st.download_button(
                        label="Download Cluster Profiles (CSV)",
                        data=csv_profiles,
                        file_name="cluster_profiles.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👈 Please upload a CSV file to begin analysis")
        
        st.markdown("""
        ### Expected CSV Format
        
        Your CSV should contain student performance data with columns including:
        - Current CGPA
        - Attendance percentage
        - Study hours
        - Social media hours
        - Probation status
        - Scholarship status
        - And other academic metrics
        
        **Sample column names:**
        - `What is your current CGPA?`
        - `Average attendance on class`
        - `How many hour do you study daily?`
        - `Did you ever fall in probation?`
        
        You can also use the sample data: `Students_Performance_data_set.csv`
        """)

if __name__ == "__main__":
    main()
