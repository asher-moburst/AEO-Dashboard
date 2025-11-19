from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend to connect

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Store processed data in memory
processed_data = {
    'brand_presence': None,
    'sources_full': None,
    'sources_detailed': None,
    'sources_combined': None,
    'perception': None,
    'citation_tracker': None,
    'events_log': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_brand_presence(file_path):
    """Process the brand presence file with raw query data"""
    import json
    from collections import Counter, defaultdict
    
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows from brand presence file")
        print(f"Columns: {df.columns.tolist()}")
        
        # Extract brand name by correlating rank to ordered_brands
        brand_name = None
        if 'rank' in df.columns and 'ordered_brands' in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row['rank']) and row['rank'] > 0 and pd.notna(row['ordered_brands']):
                    try:
                        brands = json.loads(row['ordered_brands'])
                        if len(brands) >= int(row['rank']):
                            brand_name = brands[int(row['rank']) - 1]
                            print(f"Extracted brand name: {brand_name}")
                            break
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        
        # Filter out Claude from analysis
        df = df[df['llm_name'].str.lower() != 'claude'].copy()
        print(f"After filtering out Claude: {len(df)} rows")
        
        # Normalize LLM names
        def normalize_llm_name(name):
            if pd.isna(name):
                return name
            name_lower = str(name).lower().strip()
            
            llm_mapping = {
                'aio': 'Google AIO',
                'ai overview': 'Google AIO',
                'google aio': 'Google AIO',
                'chatgpt': 'ChatGPT',
                'chat gpt': 'ChatGPT',
                'gpt': 'ChatGPT',
                'gemini': 'Gemini',
                'perplexity': 'Perplexity',
                'claude': 'Claude'
            }
            
            return llm_mapping.get(name_lower, name)
        
        df['llm_name'] = df['llm_name'].apply(normalize_llm_name)
        print(f"Normalized LLM names: {df['llm_name'].unique().tolist()}")
        
        # Convert analysis_date to datetime
        df['analysis_date'] = pd.to_datetime(df['analysis_date'])
        df['week_start'] = df['analysis_date']
        df['is_present'] = df['is_present'].astype(bool)
        
        # Calculate brand presence percentage by actual date
        weekly_trend = df.groupby('analysis_date').agg({
            'is_present': 'mean'
        }).reset_index()
        weekly_trend['brand_presence'] = weekly_trend['is_present'] * 100
        weekly_trend = weekly_trend.sort_values('analysis_date')
        weekly_trend = weekly_trend.rename(columns={'analysis_date': 'week_start'})
        
        # LLM performance over time
        llm_weekly_trend = df.groupby(['analysis_date', 'llm_name']).agg({
            'is_present': 'mean'
        }).reset_index()
        llm_weekly_trend['brand_presence'] = llm_weekly_trend['is_present'] * 100
        llm_weekly_trend = llm_weekly_trend.sort_values('analysis_date')
        llm_weekly_trend = llm_weekly_trend.rename(columns={'analysis_date': 'week_start'})
        
        # Average rank by LLM over time
        llm_rank_trend = df[df['is_present'] == True].groupby(['analysis_date', 'llm_name']).agg({
            'rank': 'mean'
        }).reset_index()
        llm_rank_trend = llm_rank_trend.sort_values('analysis_date')
        llm_rank_trend = llm_rank_trend.rename(columns={'analysis_date': 'week_start'})
        
        # Get last week and previous week data
        if len(weekly_trend) >= 2:
            last_week = weekly_trend.iloc[-1]
            prev_week = weekly_trend.iloc[-2]
            current_presence = last_week['brand_presence']
            prev_presence = prev_week['brand_presence']
            week_change = current_presence - prev_presence
            latest_date = last_week['week_start']
        else:
            current_presence = weekly_trend.iloc[-1]['brand_presence'] if len(weekly_trend) > 0 else 0
            week_change = 0
            latest_date = weekly_trend.iloc[-1]['week_start'] if len(weekly_trend) > 0 else None
        
        # LLM-specific metrics
        llm_metrics = []
        for llm in df['llm_name'].unique():
            llm_df = df[df['llm_name'] == llm]
            
            llm_trend = llm_df.groupby('analysis_date').agg({
                'is_present': 'mean'
            }).reset_index()
            
            if len(llm_trend) >= 2:
                last = llm_trend.iloc[-1]
                prev = llm_trend.iloc[-2]
                llm_current = last['is_present'] * 100
                llm_prev = prev['is_present'] * 100
                llm_change = llm_current - llm_prev
            else:
                llm_current = llm_trend.iloc[-1]['is_present'] * 100 if len(llm_trend) > 0 else 0
                llm_change = 0
            
            # Average rank when present
            present_df = llm_df[llm_df['is_present'] == True]
            avg_rank = present_df['rank'].mean() if len(present_df) > 0 else None
            
            llm_metrics.append({
                'llm': llm,
                'presence': round(llm_current, 1),
                'change': round(llm_change, 1),
                'avg_rank': round(avg_rank, 2) if avg_rank else None
            })
        
        # Topic analysis
        topic_metrics = []
        if 'topic' in df.columns:
            for topic in df['topic'].unique():
                if pd.isna(topic):
                    continue
                    
                topic_df = df[df['topic'] == topic]
                
                topic_trend = topic_df.groupby('analysis_date').agg({
                    'is_present': 'mean'
                }).reset_index()
                
                if len(topic_trend) >= 2:
                    last = topic_trend.iloc[-1]
                    prev = topic_trend.iloc[-2]
                    topic_current = last['is_present'] * 100
                    topic_prev = prev['is_present'] * 100
                    topic_change = topic_current - topic_prev
                else:
                    topic_current = topic_trend.iloc[-1]['is_present'] * 100 if len(topic_trend) > 0 else 0
                    topic_change = 0
                
                topic_metrics.append({
                    'topic': topic,
                    'presence': round(topic_current, 1),
                    'change': round(topic_change, 1)
                })
        
        # Topic by LLM analysis
        topic_by_llm_data = {}
        topic_rank_by_llm_data = {}
        
        if 'topic' in df.columns:
            for llm in df['llm_name'].unique():
                topic_by_llm_data[llm] = {}
                topic_rank_by_llm_data[llm] = {}
                
                for topic in df['topic'].unique():
                    if pd.isna(topic):
                        continue
                    
                    subset = df[(df['llm_name'] == llm) & (df['topic'] == topic)]
                    if len(subset) == 0:
                        continue
                    
                    # Presence by date
                    topic_llm_trend = subset.groupby('analysis_date').agg({
                        'is_present': 'mean'
                    }).reset_index()
                    topic_llm_trend = topic_llm_trend.sort_values('analysis_date')
                    
                    topic_by_llm_data[llm][topic] = {
                        'dates': topic_llm_trend['analysis_date'].dt.strftime('%Y-%m-%d').tolist(),
                        'presence': (topic_llm_trend['is_present'] * 100).round(1).tolist()
                    }
                    
                    # Average rank by date (only when present)
                    present_subset = subset[subset['is_present'] == True]
                    if len(present_subset) > 0:
                        rank_trend = present_subset.groupby('analysis_date').agg({
                            'rank': 'mean'
                        }).reset_index()
                        rank_trend = rank_trend.sort_values('analysis_date')
                        
                        topic_rank_by_llm_data[llm][topic] = {
                            'dates': rank_trend['analysis_date'].dt.strftime('%Y-%m-%d').tolist(),
                            'rank': rank_trend['rank'].round(2).tolist()
                        }
        
        # Prepare chart data
        chart_data = {
            'overall_trend': {
                'dates': weekly_trend['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                'presence': weekly_trend['brand_presence'].round(1).tolist()
            },
            'llm_trend': {},
            'llm_rank_trend': {}
        }
        
        for llm in df['llm_name'].unique():
            llm_data = llm_weekly_trend[llm_weekly_trend['llm_name'] == llm]
            chart_data['llm_trend'][llm] = {
                'dates': llm_data['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                'presence': llm_data['brand_presence'].round(1).tolist()
            }
            
            llm_rank_data = llm_rank_trend[llm_rank_trend['llm_name'] == llm]
            if len(llm_rank_data) > 0:
                chart_data['llm_rank_trend'][llm] = {
                    'dates': llm_rank_data['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                    'rank': llm_rank_data['rank'].round(2).tolist()
                }
        
        return {
            'success': True,
            'brand_name': brand_name,
            'current_presence': round(current_presence, 1),
            'week_change': round(week_change, 1),
            'latest_date': latest_date.strftime('%Y-%m-%d') if latest_date else None,
            'llm_metrics': llm_metrics,
            'topic_metrics': topic_metrics,
            'chart_data': chart_data,
            'topic_by_llm_data': topic_by_llm_data,
            'topic_rank_by_llm_data': topic_rank_by_llm_data,
            'available_dates': weekly_trend['week_start'].dt.strftime('%Y-%m-%d').tolist()
        }
        
    except Exception as e:
        print(f"Error processing brand presence: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def process_sources_file(file_path, file_type='full'):
    """Process sources export files"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows from sources file ({file_type})")
        
        # Normalize LLM names
        def normalize_llm_name(name):
            if pd.isna(name):
                return name
            name_lower = str(name).lower().strip()
            llm_mapping = {
                'aio': 'Google AIO',
                'google aio': 'Google AIO',
                'chatgpt': 'ChatGPT',
                'gemini': 'Gemini',
                'perplexity': 'Perplexity'
            }
            return llm_mapping.get(name_lower, name)
        
        if 'llm_name' in df.columns:
            df['llm_name'] = df['llm_name'].apply(normalize_llm_name)
        
        # Convert date column
        if 'analysis_date' in df.columns:
            df['analysis_date'] = pd.to_datetime(df['analysis_date'])
        
        # Source type distribution
        source_dist = {}
        if 'source_type' in df.columns:
            source_counts = df['source_type'].value_counts()
            source_dist = {
                'labels': source_counts.index.tolist(),
                'values': source_counts.values.tolist()
            }
        
        # LLM distribution
        llm_dist = {}
        if 'llm_name' in df.columns:
            llm_counts = df['llm_name'].value_counts()
            llm_dist = {
                'labels': llm_counts.index.tolist(),
                'values': llm_counts.values.tolist()
            }
        
        # Top sources
        top_sources = []
        if 'source' in df.columns:
            source_counts = df['source'].value_counts().head(10)
            top_sources = [
                {'source': src, 'count': int(count)}
                for src, count in source_counts.items()
            ]
        
        return {
            'success': True,
            'total_sources': len(df),
            'source_distribution': source_dist,
            'llm_distribution': llm_dist,
            'top_sources': top_sources
        }
        
    except Exception as e:
        print(f"Error processing sources: {str(e)}")
        return {'success': False, 'error': str(e)}

def process_citation_tracker(file_path):
    """Process citation tracker file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows from citation tracker")
        
        # Calculate citation metrics
        total_citations = len(df)
        
        # If there's a date column, calculate trend
        citation_trend = {}
        if 'date' in df.columns or 'analysis_date' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'analysis_date'
            df[date_col] = pd.to_datetime(df[date_col])
            
            daily_citations = df.groupby(date_col).size().reset_index(name='count')
            daily_citations = daily_citations.sort_values(date_col)
            
            citation_trend = {
                'dates': daily_citations[date_col].dt.strftime('%Y-%m-%d').tolist(),
                'counts': daily_citations['count'].tolist()
            }
            
            # Latest week citations
            if len(daily_citations) > 0:
                latest_date = daily_citations[date_col].max()
                week_ago = latest_date - timedelta(days=7)
                latest_citations = len(df[df[date_col] >= week_ago])
            else:
                latest_citations = 0
        else:
            latest_citations = 0
        
        return {
            'success': True,
            'total_citations': total_citations,
            'latest_citations': latest_citations,
            'citation_trend': citation_trend
        }
        
    except Exception as e:
        print(f"Error processing citation tracker: {str(e)}")
        return {'success': False, 'error': str(e)}

def process_perception(file_path):
    """Process brand perception file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows from perception data")
        
        # Calculate overall perception score
        if 'score' in df.columns:
            overall_score = df['score'].mean()
        else:
            overall_score = None
        
        # Get attributes with scores
        attributes = []
        if 'attribute' in df.columns and 'score' in df.columns:
            attr_scores = df.groupby('attribute')['score'].mean().reset_index()
            attr_scores = attr_scores.sort_values('score', ascending=False)
            
            for _, row in attr_scores.iterrows():
                attributes.append({
                    'attribute': row['attribute'],
                    'score': round(row['score'], 2),
                    'change': 0  # Would need historical data to calculate
                })
        
        return {
            'success': True,
            'overall_score': round(overall_score, 2) if overall_score else None,
            'attributes': attributes
        }
        
    except Exception as e:
        print(f"Error processing perception data: {str(e)}")
        return {'success': False, 'error': str(e)}

def process_events_log(file_path):
    """Process events log file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows from events log")
        
        # Get recent events
        events = []
        if 'date' in df.columns and 'description' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            
            for _, row in df.head(20).iterrows():  # Get last 20 events
                events.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'description': row['description']
                })
        
        return {
            'success': True,
            'events': events
        }
        
    except Exception as e:
        print(f"Error processing events log: {str(e)}")
        return {'success': False, 'error': str(e)}

# API Routes
@app.route('/')
def home():
    return jsonify({
        'message': 'AEO Dashboard API',
        'status': 'running',
        'endpoints': {
            'upload': '/upload',
            'data': '/data',
            'reset': '/reset'
        }
    })

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        results = {}
        
        # Process each file type
        file_types = {
            'brand_presence': process_brand_presence,
            'sources_full': lambda f: process_sources_file(f, 'full'),
            'sources_detailed': lambda f: process_sources_file(f, 'detailed'),
            'citation_tracker': process_citation_tracker,
            'perception': process_perception,
            'events_log': process_events_log
        }
        
        for file_key, process_func in file_types.items():
            if file_key in request.files:
                file = request.files[file_key]
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    # Process the file
                    result = process_func(filepath)
                    processed_data[file_key] = result
                    results[file_key] = {'uploaded': True, 'processed': result.get('success', False)}
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/data', methods=['GET'])
def get_data():
    """Return all processed data"""
    return jsonify({
        'success': True,
        'data': processed_data
    })

@app.route('/reset', methods=['POST'])
def reset_data():
    """Reset all processed data"""
    global processed_data
    processed_data = {
        'brand_presence': None,
        'sources_full': None,
        'sources_detailed': None,
        'sources_combined': None,
        'perception': None,
        'citation_tracker': None,
        'events_log': None
    }
    return jsonify({'success': True, 'message': 'Data reset successfully'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
