package org.lemming.outputs;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.swing.Timer;

import org.jzy3d.chart.Chart;
import org.jzy3d.chart.ChartLauncher;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRainbow;
import org.jzy3d.maths.Coord3d;
import org.jzy3d.plot3d.primitives.MultiColorScatterList;
import org.lemming.data.Localization;
import org.lemming.data.Rendering;
import org.lemming.input.SI;

public class Jzy3dScatterplot extends SI<Localization> implements Rendering {

	Chart chart;
	MultiColorScatterList scatter;
	
	List<Coord3d> coordList = Collections.synchronizedList(new ArrayList<Coord3d>());
	
	@Override
	public void beforeRun() {
		scatter = new MultiColorScatterList( coordList , new ColorMapper( new ColorMapRainbow(), -0.5f, 0.5f ) );
		
		chart = new Chart();
		
		new Timer(100, 
				new ActionListener() {
					@Override
					public void actionPerformed(ActionEvent e) {
						update(); 
					}
				}).start();
	}
	
	@Override
	public void process(Localization l) {
		coordList.add(new Coord3d(l.getX(),l.getY(),0));
	}

	@Override
	public void afterRun() {
		chart.getScene().add(scatter);
		
		ChartLauncher.openChart(chart);
	}
	
	void update() {
		chart.render();
	}
}
