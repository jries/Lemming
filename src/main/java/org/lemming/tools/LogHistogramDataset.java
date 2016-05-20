package org.lemming.tools;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jfree.chart.util.ParamChecks;
import org.jfree.data.statistics.HistogramType;
import org.jfree.data.xy.AbstractIntervalXYDataset;
import org.jfree.util.PublicCloneable;

/**
 * A HistogramDataset that returns the log of the count in each bin (plus one),
 * so as to have a logarithmic plot.
 */
public class LogHistogramDataset extends AbstractIntervalXYDataset implements PublicCloneable {
	
	private static final long serialVersionUID = 6012084169414194555L;
	private final List<Map<String,Object>> list;
	private HistogramType type;
	
	public LogHistogramDataset(){
		this.list = new ArrayList<>();
		this.type = HistogramType.FREQUENCY;
	}
	
	public void setType(HistogramType type) {
        ParamChecks.nullNotPermitted(type, "type");
        this.type = type;
        fireDatasetChanged();
    }
	
	public void addSeries(String key, int[] counts, int bins, double d, double e){
		
		double binWidth = (e - d) / bins;
		List<LogHistogramBin> binList = new ArrayList<>(bins);
		double lower = d;
        double upper;
        if (counts.length!=bins) return;
		for (int i = 0; i < bins; i++) {
	            LogHistogramBin bin;
	            // make sure bins[bins.length]'s upper boundary ends at maximum
	            // to avoid the rounding issue. the bins[0] lower boundary is
	            // guaranteed start from min
	            if (i == bins - 1) {
	                bin = new LogHistogramBin(lower, e);
	            }
	            else {
	                upper = d + (i + 1) * binWidth;
	                bin = new LogHistogramBin(lower, upper);
	                lower = upper;
	            }
	            bin.setCount(counts[i]);
	            binList.add(bin);
	    }
		
	    // generic map for each series
        Map<String,Object> map = new HashMap<>();
        map.put("key", key);
        map.put("bins", binList);
        map.put("values.length", bins);
        map.put("bin width", binWidth);
        this.list.add(map);
        fireDatasetChanged();
	}
	
	@Override
	public Number getY(int series, int item) {
		List<LogHistogramBin> bins = getBins(series);
		LogHistogramBin bin = bins.get(item);
        double total = getTotal(series);
        double binWidth = getBinWidth(series);

        if (this.type == HistogramType.FREQUENCY) {
            return Math.log(1 + bin.getCount());
        }
        else if (this.type == HistogramType.RELATIVE_FREQUENCY) {
            return bin.getCount() / total;
        }
        else if (this.type == HistogramType.SCALE_AREA_TO_1) {
            return bin.getCount() / (binWidth * total);
        }
        else { // pretty sure this shouldn't ever happen
            throw new IllegalStateException();
        }
	}

	@Override
	public int getItemCount(int series) {
		return getBins(series).size();
	}

	@Override
	public Number getX(int series, int item) {
		 List<LogHistogramBin> bins = getBins(series);
		 LogHistogramBin bin = bins.get(item);
		return (bin.getStartBoundary() + bin.getEndBoundary()) / 2.;
	}

	@Override
	public Number getStartX(int series, int item) {
		List<LogHistogramBin> bins = getBins(series);
		LogHistogramBin bin = bins.get(item);
        return bin.getStartBoundary();

	}

	@Override
	public Number getEndX(int series, int item) {
		List<LogHistogramBin> bins = getBins(series);
		LogHistogramBin bin = bins.get(item);
        return bin.getEndBoundary();
	}

	@Override
	public Number getStartY(int series, int item) {
		 return getY(series, item);
	}

	@Override
	public Number getEndY(int series, int item) {
		return getY(series, item);
	}

	@Override
	public int getSeriesCount() {
		return this.list.size();
	}

	@Override
	public String getSeriesKey(int series) {
		Map<String, Object> map = this.list.get(series);
		return (String) map.get("key");
	}
	
	private int getTotal(int series) {
		Map<String, Object> map = this.list.get(series);
        return (Integer) map.get("values.length");
	}
	
	@SuppressWarnings("unchecked")
	private List<LogHistogramBin> getBins(int series) {
		Map<String, Object> map = this.list.get(series);
		return (List<LogHistogramBin>) map.get("bins");
	}
	
	 private double getBinWidth(int series) {
		Map<String, Object> map = this.list.get(series);
		return (Double) map.get("bin width");
    }

}
