package org.lemming.processors;

import ij.gui.Roi;

import org.lemming.interfaces.Localization;

/**
 * @author Ronny Sczech
 *
 */
public class ROISelectProcessor extends SingleInputSingleOutput<Localization,Localization> {	
	
	private Roi roi;
	private boolean hasMoreOutputs = true;
	
	/**
	 * @param x - x
	 * @param xLength - width
	 * @param y - y
	 * @param yLength - height
	 */
	public ROISelectProcessor(int x, int xLength, int y, int yLength) {
		roi = new Roi(x,y,xLength,yLength);
	}
	
	/**
	 * @param theRoi - Region of interest
	 */
	public ROISelectProcessor(Roi theRoi) {
		roi = theRoi;
	}
	
	@Override
	public void process(Localization loc) {
		if (loc==null) return;
		if(loc.isLast()){ 
			hasMoreOutputs = false;
			stop();
			return;
		}
		double x = loc.getX();
		double y = loc.getY();
		
		if (roi.contains( (int) x, (int) y))
			output.put(loc);
	}

	@Override
	public boolean hasMoreOutputs() {
		return hasMoreOutputs ;
	}

}
