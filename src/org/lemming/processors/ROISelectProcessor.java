package org.lemming.processors;

import ij.gui.Roi;

import org.lemming.data.Localization;
import org.lemming.interfaces.Processor;

public class ROISelectProcessor implements Processor<Localization,Localization> {	
	
	Roi roi;
	
	public ROISelectProcessor(int x, int xLength, int y, int yLength) {
		roi = new Roi(x,y,xLength,yLength);
	}
	
	public ROISelectProcessor(Roi theRoi) {
		roi = theRoi;
	}
	
	@Override
	public Localization process(Localization loc) {
                double x = loc.getX();
                double y = loc.getY();
                
                if (roi.contains( (int) x, (int) y)) {
                        return loc;
                } else {
                        return null;
                }
	}

}
