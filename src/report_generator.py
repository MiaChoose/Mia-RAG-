"""
报告生成器
生成可视化报告和HTML报告
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, results_dir: str):
        """
        初始化报告生成器
        
        Args:
            results_dir: 结果目录
        """
        self.results_dir = Path(results_dir)
        
        # 加载评估结果
        self.df_results = None
        self.comparison = None
        self.report = None
        
        self._load_results()
    
    def _load_results(self):
        """加载结果文件"""
        try:
            # 加载评估结果
            csv_file = self.results_dir / "evaluation_results.csv"
            if csv_file.exists():
                self.df_results = pd.read_csv(csv_file)
                logger.info(f"加载评估结果: {len(self.df_results)} 条")
            
            # 加载对比结果
            comparison_file = self.results_dir / "search_type_comparison.csv"
            if comparison_file.exists():
                self.comparison = pd.read_csv(comparison_file, header=[0, 1], index_col=0)
                logger.info(f"加载对比结果: {len(self.comparison)} 种配置")
            
            # 加载测试报告
            report_file = self.results_dir / "test_report.json"
            if report_file.exists():
                with open(report_file, 'r', encoding='utf-8') as f:
                    self.report = json.load(f)
                logger.info("加载测试报告")
        
        except Exception as e:
            logger.error(f"加载结果文件失败: {e}")
    
    def generate_visualizations(self) -> List[str]:
        """
        生成可视化图表
        
        Returns:
            List[str]: 生成的图表文件路径列表
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import seaborn as sns
            sns.set_style("whitegrid")
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError:
            logger.error("需要安装 matplotlib 和 seaborn: pip install matplotlib seaborn")
            return []
        
        generated_files = []
        
        if self.df_results is None or len(self.df_results) == 0:
            logger.warning("没有评估结果可供可视化")
            return generated_files
        
        try:
            # 1. 指标对比雷达图
            radar_file = self._generate_radar_chart(plt, sns)
            if radar_file:
                generated_files.append(radar_file)
            
            # 2. 指标箱线图
            box_file = self._generate_box_plot(plt, sns)
            if box_file:
                generated_files.append(box_file)
            
            # 3. 响应时间对比
            time_file = self._generate_response_time_chart(plt, sns)
            if time_file:
                generated_files.append(box_file)
            
            # 4. 指标热力图
            heatmap_file = self._generate_heatmap(plt, sns)
            if heatmap_file:
                generated_files.append(heatmap_file)
            
            logger.info(f"生成了 {len(generated_files)} 个可视化图表")
            
        except Exception as e:
            logger.error(f"生成可视化图表失败: {e}")
        
        return generated_files
    
    def _generate_radar_chart(self, plt, sns) -> str:
        """生成雷达图"""
        if 'search_type' not in self.df_results.columns:
            return None
        
        try:
            import numpy as np
            
            metrics = ['context_precision', 'context_recall', 'faithfulness', 
                      'answer_relevancy', 'answer_correctness', 'answer_similarity']
            available_metrics = [m for m in metrics if m in self.df_results.columns]
            
            if len(available_metrics) < 3:
                return None
            
            # 计算每种检索类型的平均值
            grouped = self.df_results.groupby('search_type')[available_metrics].mean()
            
            # 雷达图
            angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            for search_type in grouped.index:
                values = grouped.loc[search_type].tolist()
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=search_type)
                ax.fill(angles, values, alpha=0.15)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(available_metrics)
            ax.set_ylim(0, 1)
            ax.set_title('检索方式指标对比 (雷达图)', size=16, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
            
            file_path = self.results_dir / "radar_chart.png"
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"雷达图已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"生成雷达图失败: {e}")
            return None
    
    def _generate_box_plot(self, plt, sns) -> str:
        """生成箱线图"""
        if 'search_type' not in self.df_results.columns:
            return None
        
        try:
            metrics = ['context_precision', 'context_recall', 'faithfulness', 
                      'answer_relevancy', 'answer_correctness', 'answer_similarity']
            available_metrics = [m for m in metrics if m in self.df_results.columns]
            
            if len(available_metrics) < 2:
                return None
            
            # 准备数据
            df_melted = self.df_results.melt(
                id_vars=['search_type'],
                value_vars=available_metrics,
                var_name='metric',
                value_name='score'
            )
            
            # 移除None值
            df_melted = df_melted.dropna(subset=['score'])
            
            if len(df_melted) == 0:
                logger.warning("箱线图：所有数据为空，跳过生成")
                return None
            
            # 绘制箱线图
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.boxplot(data=df_melted, x='metric', y='score', hue='search_type', ax=ax)
            
            ax.set_title('各指标得分分布 (箱线图)', size=16)
            ax.set_xlabel('指标', size=12)
            ax.set_ylabel('得分', size=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(title='检索方式', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            file_path = self.results_dir / "box_plot.png"
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"箱线图已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"生成箱线图失败: {e}")
            return None
    
    def _generate_response_time_chart(self, plt, sns) -> str:
        """生成响应时间对比图"""
        if 'response_time' not in self.df_results.columns or 'search_type' not in self.df_results.columns:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            grouped = self.df_results.groupby('search_type')['response_time'].agg(['mean', 'std'])
            grouped.plot(kind='bar', y='mean', yerr='std', ax=ax, legend=False)
            
            ax.set_title('平均响应时间对比', size=16)
            ax.set_xlabel('检索方式', size=12)
            ax.set_ylabel('响应时间 (秒)', size=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            file_path = self.results_dir / "response_time.png"
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"响应时间图已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"生成响应时间图失败: {e}")
            return None
    
    def _generate_heatmap(self, plt, sns) -> str:
        """生成热力图"""
        if self.comparison is None or len(self.comparison) == 0:
            return None
        
        try:
            # 提取 mean 值
            mean_data = self.comparison.xs('mean', level=1, axis=1)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(mean_data.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                       cbar_kws={'label': '平均得分'})
            
            ax.set_title('检索方式 × 指标热力图', size=16)
            ax.set_xlabel('检索方式', size=12)
            ax.set_ylabel('指标', size=12)
            
            file_path = self.results_dir / "heatmap.png"
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"热力图已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"生成热力图失败: {e}")
            return None
    
    def _load_evaluation_results(self) -> List[Dict[str, Any]]:
        """加载详细评估结果"""
        try:
            json_file = self.results_dir / "evaluation_results.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    logger.info(f"加载了 {len(results)} 条详细评估结果")
                    return results
        except Exception as e:
            logger.error(f"加载评估结果失败: {e}")
        return []
    
    def generate_html_report(self) -> str:
        """
        生成 HTML 报告
        
        Returns:
            str: HTML 文件路径
        """
        if self.df_results is None or self.report is None:
            logger.warning("缺少必要数据，无法生成 HTML 报告")
            return None
        
        try:
            html_content = self._build_html_content()
            
            file_path = self.results_dir / "report.html"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML 报告已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"生成 HTML 报告失败: {e}")
            return None
    
    def _build_html_content(self) -> str:
        """构建 HTML 内容"""
        summary = self.report.get('test_summary', {})
        stats = self.report.get('summary_statistics', {})
        
        # 加载详细评估结果
        evaluation_results = self._load_evaluation_results()
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG 测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-left: 4px solid #4CAF50; padding-left: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }}
        .summary-card h3 {{ margin: 0; font-size: 14px; opacity: 0.9; }}
        .summary-card p {{ margin: 10px 0 0 0; font-size: 32px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
        th, td {{ padding: 10px 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; position: sticky; top: 0; z-index: 10; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric-good {{ color: #4CAF50; font-weight: bold; }}
        .metric-medium {{ color: #FF9800; font-weight: bold; }}
        .metric-poor {{ color: #f44336; font-weight: bold; }}
        .metric-na {{ color: #999; font-style: italic; }}
        .chart {{ margin: 20px 0; text-align: center; }}
        .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #888; }}
        
        /* 详细数据表格样式 */
        .data-controls {{ display: flex; justify-content: space-between; align-items: center; margin: 20px 0; flex-wrap: wrap; gap: 15px; }}
        .search-box input {{ padding: 10px 15px; width: 300px; border: 2px solid #ddd; border-radius: 5px; font-size: 14px; }}
        .search-box input:focus {{ outline: none; border-color: #4CAF50; }}
        .filter-controls {{ display: flex; gap: 15px; align-items: center; }}
        .filter-controls label {{ font-size: 14px; color: #555; }}
        .filter-controls select {{ padding: 8px 12px; border: 2px solid #ddd; border-radius: 5px; font-size: 14px; cursor: pointer; }}
        .table-container {{ overflow-x: auto; max-height: 600px; border: 1px solid #ddd; border-radius: 5px; }}
        .text-cell {{ max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: help; }}
        .detail-btn {{ background-color: #2196F3; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; }}
        .detail-btn:hover {{ background-color: #0b7dda; }}
        
        /* 分页样式 */
        .pagination {{ display: flex; justify-content: center; align-items: center; gap: 15px; margin: 20px 0; }}
        .pagination button {{ padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; }}
        .pagination button:hover:not(:disabled) {{ background-color: #45a049; }}
        .pagination button:disabled {{ background-color: #ccc; cursor: not-allowed; }}
        .pagination span {{ font-size: 14px; color: #555; }}
        
        /* 模态框样式 */
        .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.5); }}
        .modal-content {{ background-color: #fefefe; margin: 3% auto; padding: 20px 30px; border: 1px solid #888; border-radius: 8px; width: 80%; max-width: 900px; max-height: 85vh; overflow-y: auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .close {{ color: #aaa; float: right; font-size: 32px; font-weight: bold; cursor: pointer; line-height: 20px; }}
        .close:hover, .close:focus {{ color: #000; }}
        .detail-section {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .detail-section h3 {{ margin-top: 0; color: #333; font-size: 16px; }}
        .detail-section p {{ margin: 10px 0; color: #555; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; }}
        .context-item {{ margin: 15px 0; padding: 0; background-color: white; border-radius: 6px; border: 1px solid #ddd; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .context-header {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
        .context-header strong {{ font-size: 14px; }}
        .context-length {{ font-size: 12px; opacity: 0.9; background-color: rgba(255,255,255,0.2); padding: 3px 8px; border-radius: 10px; }}
        .context-text {{ padding: 15px; max-height: 300px; overflow-y: auto; line-height: 1.8; color: #333; font-size: 14px; text-align: justify; }}
        .context-text::-webkit-scrollbar {{ width: 8px; }}
        .context-text::-webkit-scrollbar-track {{ background: #f1f1f1; border-radius: 4px; }}
        .context-text::-webkit-scrollbar-thumb {{ background: #888; border-radius: 4px; }}
        .context-text::-webkit-scrollbar-thumb:hover {{ background: #555; }}
        .metrics-table {{ width: 100%; background-color: white; }}
        .metrics-table td {{ padding: 8px; border-bottom: 1px solid #eee; }}
        .metrics-table td:first-child {{ font-weight: bold; color: #555; width: 40%; }}
        .metrics-table td:last-child {{ color: #333; text-align: right; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 RAG 系统测试报告</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>总测试数</h3>
                <p>{summary.get('total_tests', 0)}</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>成功测试</h3>
                <p>{summary.get('successful_tests', 0)}</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>测试问题数</h3>
                <p>{summary.get('questions', 0)}</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <h3>检索配置数</h3>
                <p>{summary.get('test_configs', 0)}</p>
            </div>
        </div>
        
        <h2>整体指标统计</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>平均值</th>
                <th>标准差</th>
                <th>最小值</th>
                <th>最大值</th>
                <th>中位数</th>
            </tr>
"""
        
        for metric, values in stats.items():
            if metric != 'overall_score' and isinstance(values, dict):
                mean_val = values.get('mean', 0)
                if mean_val is None:
                    mean_val = 0
                metric_class = 'metric-good' if mean_val >= 0.7 else ('metric-medium' if mean_val >= 0.5 else 'metric-poor')
                
                # 安全地获取值，处理None
                def safe_format(val):
                    return f"{val:.3f}" if val is not None else "N/A"
                
                html += f"""
            <tr>
                <td>{metric}</td>
                <td class="{metric_class}">{safe_format(mean_val)}</td>
                <td>{safe_format(values.get('std'))}</td>
                <td>{safe_format(values.get('min'))}</td>
                <td>{safe_format(values.get('max'))}</td>
                <td>{safe_format(values.get('median'))}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>可视化图表</h2>
"""
        
        # 添加图表（检查是否存在）
        chart_files = {
            'radar_chart.png': '指标对比雷达图',
            'box_plot.png': '指标分布箱线图',
            'heatmap.png': '检索方式对比热力图',
            'response_time.png': '响应时间对比'}
        
        has_charts = False
        for img_file, title in chart_files.items():
            if (self.results_dir / img_file).exists():
                has_charts = True
                html += f"""
        <div class="chart">
            <h3 style="color: #555; margin-bottom: 10px;">{title}</h3>
            <img src="{img_file}" alt="{title}">
        </div>
"""
        
        if not has_charts:
            html += """
        <div style="padding: 20px; background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; color: #856404;">
            <p>⚠️ 暂无可视化图表。图表生成需要安装 matplotlib 和 seaborn:</p>
            <code style="background-color: #f8f9fa; padding: 5px 10px; border-radius: 3px; display: inline-block; margin-top: 5px;">
                pip install matplotlib seaborn
            </code>
        </div>
"""
        
        html += """
        <h2>📋 详细测试数据</h2>
        <div class="data-controls">
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="🔍 搜索问题、答案或配置..." onkeyup="filterTable()">
            </div>
            <div class="filter-controls">
                <label>检索类型：
                    <select id="searchTypeFilter" onchange="filterTable()">
                        <option value="">全部</option>
"""
        
        # 添加检索类型选项
        if evaluation_results:
            search_types = set(r.get('search_type', '') for r in evaluation_results)
            for st in sorted(search_types):
                html += f'                        <option value="{st}">{st}</option>\n'
        
        html += """                    </select>
                </label>
                <label>每页显示：
                    <select id="pageSize" onchange="changePageSize()">
                        <option value="10">10</option>
                        <option value="20" selected>20</option>
                        <option value="50">50</option>
                        <option value="100">100</option>
                    </select>
                </label>
            </div>
        </div>
        
        <div class="table-container">
            <table id="detailTable">
                <thead>
                    <tr>
                        <th>序号</th>
                        <th onclick="sortTable(1)" style="cursor:pointer;">问题 ▼</th>
                        <th>答案</th>
                        <th>标准答案</th>
                        <th onclick="sortTable(4)" style="cursor:pointer;">检索类型 ▼</th>
                        <th onclick="sortTable(5)" style="cursor:pointer;">上下文精确度 ▼</th>
                        <th onclick="sortTable(6)" style="cursor:pointer;">上下文召回率 ▼</th>
                        <th onclick="sortTable(7)" style="cursor:pointer;">忠实度 ▼</th>
                        <th onclick="sortTable(8)" style="cursor:pointer;">答案相关性 ▼</th>
                        <th onclick="sortTable(9)" style="cursor:pointer;">答案正确性 ▼</th>
                        <th onclick="sortTable(10)" style="cursor:pointer;">答案相似度 ▼</th>
                        <th onclick="sortTable(11)" style="cursor:pointer;">响应时间(s) ▼</th>
                        <th>操作</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
"""
        
        # 添加表格数据
        for idx, result in enumerate(evaluation_results, 1):
            question = result.get('question', '')
            response = result.get('answer', '')   #要改
            reference = result.get('reference', '')
            search_type = result.get('ground_truth', '')   #要改
            
            # 评估指标
            context_precision = result.get('context_precision', 0)
            context_recall = result.get('context_recall', 0)
            faithfulness = result.get('faithfulness', 0)
            answer_relevancy = result.get('answer_relevancy', 0)
            answer_correctness = result.get('answer_correctness', 0)
            answer_similarity = result.get('answer_similarity', 0)
            response_time = result.get('response_time', 0)
            
            # 缩短显示文本
            question_short = question[:50] + '...' if len(question) > 50 else question
            response_short = response[:80] + '...' if len(response) > 80 else response
            reference_short = reference[:80] + '...' if len(reference) > 80 else reference
            
            # 根据得分设置颜色类
            def get_metric_class(score):
                if score is None:
                    return 'metric-na'
                if score >= 0.7:
                    return 'metric-good'
                elif score >= 0.5:
                    return 'metric-medium'
                else:
                    return 'metric-poor'
            
            # 格式化指标显示
            def format_metric(value):
                return f"{value:.3f}" if value is not None else "N/A"
            
            def format_time(value):
                return f"{value:.2f}" if value is not None else "N/A"
            
            html += f"""
                    <tr data-index="{idx}">
                        <td>{idx}</td>
                        <td class="text-cell" title="{question}">{question_short}</td>
                        <td class="text-cell" title="{response}">{response_short}</td>
                        <td class="text-cell" title="{reference}">{reference_short}</td>
                        <td>{search_type}</td>
                        <td class="{get_metric_class(context_precision)}">{format_metric(context_precision)}</td>
                        <td class="{get_metric_class(context_recall)}">{format_metric(context_recall)}</td>
                        <td class="{get_metric_class(faithfulness)}">{format_metric(faithfulness)}</td>
                        <td class="{get_metric_class(answer_relevancy)}">{format_metric(answer_relevancy)}</td>
                        <td class="{get_metric_class(answer_correctness)}">{format_metric(answer_correctness)}</td>
                        <td class="{get_metric_class(answer_similarity)}">{format_metric(answer_similarity)}</td>
                        <td>{format_time(response_time)}</td>
                        <td><button class="detail-btn" onclick="showDetail({idx})">详情</button></td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
        
        <div class="pagination">
            <button onclick="previousPage()" id="prevBtn">« 上一页</button>
            <span id="pageInfo"></span>
            <button onclick="nextPage()" id="nextBtn">下一页 »</button>
        </div>
        
        <!-- 详情弹窗 -->
        <div id="detailModal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeDetail()">&times;</span>
                <div id="detailContent"></div>
            </div>
        </div>
        
        <script>
        // 存储所有数据
        const allData = """ + json.dumps(evaluation_results, ensure_ascii=False) + """;
        let currentPage = 1;
        let pageSize = 20;
        let filteredData = allData;
        
        // 初始化
        displayPage();
        
        function filterTable() {
            const searchText = document.getElementById('searchInput').value.toLowerCase();
            const searchType = document.getElementById('searchTypeFilter').value;
            
            filteredData = allData.filter(item => {
                const matchSearch = !searchText || 
                    item.question.toLowerCase().includes(searchText) ||
                    item.response.toLowerCase().includes(searchText) ||
                    item.config_name.toLowerCase().includes(searchText);
                
                const matchType = !searchType || item.search_type === searchType;
                
                return matchSearch && matchType;
            });
            
            currentPage = 1;
            displayPage();
        }
        
        function changePageSize() {
            pageSize = parseInt(document.getElementById('pageSize').value);
            currentPage = 1;
            displayPage();
        }
        
        function displayPage() {
            const totalPages = Math.ceil(filteredData.length / pageSize);
            const start = (currentPage - 1) * pageSize;
            const end = start + pageSize;
            
            // 隐藏所有行
            const rows = document.querySelectorAll('#tableBody tr');
            rows.forEach(row => row.style.display = 'none');
            
            // 显示当前页的行
            const displayData = filteredData.slice(start, end);
            displayData.forEach(item => {
                const index = item.test_metadata?.test_id || allData.indexOf(item) + 1;
                const row = document.querySelector(`tr[data-index="${index}"]`);
                if (row) row.style.display = '';
            });
            
            // 更新分页信息
            document.getElementById('pageInfo').textContent = 
                `第 ${currentPage} / ${totalPages || 1} 页 (共 ${filteredData.length} 条记录)`;
            
            // 更新按钮状态
            document.getElementById('prevBtn').disabled = currentPage === 1;
            document.getElementById('nextBtn').disabled = currentPage >= totalPages;
        }
        
        function previousPage() {
            if (currentPage > 1) {
                currentPage--;
                displayPage();
            }
        }
        
        function nextPage() {
            const totalPages = Math.ceil(filteredData.length / pageSize);
            if (currentPage < totalPages) {
                currentPage++;
                displayPage();
            }
        }
        
        function sortTable(columnIndex) {
            // 简单的排序功能
            alert('排序功能开发中...');
        }
        
        function showDetail(index) {
            const data = allData[index - 1];
            if (!data) return;
            
            // 文本清理函数：移除多余空格和换行符
            function cleanText(text) {
                if (!text) return '';
                return text
                    .replace(/\\n+/g, ' ')           // 将换行符替换为空格
                    .replace(/\s+/g, ' ')            // 将多个空格合并为一个
                    .replace(/\s*,\s*/g, ', ')       // 规范化逗号周围的空格
                    .trim();                         // 移除首尾空格
            }
            
            const contexts = data.contexts; //要改
            const contextHtml = contexts.map((ctx, i) => {
                const cleanedText = cleanText(ctx);
                return `<div class="context-item">
                    <div class="context-header">
                        <strong>📄 上下文 ${i+1}</strong>
                        <span class="context-length">${cleanedText.length} 字符</span>
                    </div>
                    <div class="context-text">${cleanedText}</div>
                </div>`;
            }).join('');
            
            const html = `
                <h2>测试详情 #${index}</h2>
                <div class="detail-section">
                    <h3>📝 问题</h3>
                    <p>${cleanText(data.question)}</p>
                </div>
                <div class="detail-section">
                    <h3>💬 AI回答</h3>
                    <p>${cleanText(data.answer)}</p>  //要改
                </div>
                <div class="detail-section">
                    <h3>✅ 标准答案</h3>
                    <p>${cleanText(data.ground_truth) || '无'}</p>  //要改
                </div>
                <div class="detail-section">
                    <h3>📚 AI召回上下文 (${contexts.length} 个)</h3>
                    ${contextHtml || '<p>无上下文</p>'}
                </div>
                ${data.ground_truth_contexts && data.ground_truth_contexts.length > 0 ? `
                <div class="detail-section">
                    <h3>✅ 标准参考上下文 (${data.ground_truth_contexts.length} 个)</h3>
                    ${data.ground_truth_contexts.map((ctx, i) => {
                        const cleanedText = cleanText(ctx);
                        return `<div class="context-item context-truth">
                            <div class="context-header" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                                <strong>✓ 标准上下文 ${i+1}</strong>
                                <span class="context-length">${cleanedText.length} 字符</span>
                            </div>
                            <div class="context-text">${cleanedText}</div>
                        </div>`;
                    }).join('')}
                </div>
                ` : ''}
                <div class="detail-section">
                    <h3>📊 评估指标</h3>
                    <table class="metrics-table">
                        <tr><td>上下文精确度</td><td>${(data.context_precision || 0).toFixed(3)}</td></tr>
                        <tr><td>上下文召回率</td><td>${(data.context_recall || 0).toFixed(3)}</td></tr>
                        <tr><td>忠实度</td><td>${(data.faithfulness || 0).toFixed(3)}</td></tr>
                        <tr><td>答案相关性</td><td>${(data.answer_relevancy || 0).toFixed(3)}</td></tr>
                        <tr><td>答案正确性</td><td>${(data.answer_correctness || 0).toFixed(3)}</td></tr>
                        <tr><td>答案相似度</td><td>${(data.answer_similarity || 0).toFixed(3)}</td></tr>
                        <tr><td>响应时间</td><td>${(data.response_time || 0).toFixed(2)}s</td></tr>
                    </table>
                </div>
                <div class="detail-section">
                    <h3>⚙️ 配置信息</h3>
                    <p><strong>检索类型:</strong> ${data.search_type || '未知'}</p>
                    <p><strong>配置名称:</strong> ${data.config_name || '未知'}</p>
                    <p><strong>向量权重:</strong> ${data.vector_weight || 'N/A'}</p>
                    <p><strong>BM25权重:</strong> ${data.bm25_weight || 'N/A'}</p>
                </div>
            `;
            
            document.getElementById('detailContent').innerHTML = html;
            document.getElementById('detailModal').style.display = 'block';
        }
        
        function closeDetail() {
            document.getElementById('detailModal').style.display = 'none';
        }
        
        // 点击模态框外部关闭
        window.onclick = function(event) {
            const modal = document.getElementById('detailModal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
        </script>
        
        <div class="footer">
            <p>RAG 测试系统 | 生成时间: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html


# 示例用法
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 假设有结果目录
    results_dir = "results/20240101_120000"
    
    generator = ReportGenerator(results_dir)
    
    # 生成可视化
    viz_files = generator.generate_visualizations()
    print(f"生成了 {len(viz_files)} 个可视化图表")
    
    # 生成 HTML 报告
    html_file = generator.generate_html_report()
    print(f"HTML 报告: {html_file}")






