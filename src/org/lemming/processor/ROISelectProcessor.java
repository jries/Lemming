package org.lemming.processor;

import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.interfaces.Processor; 
import org.lemming.outputs.NullStoreWarning;

public class ROISelectProcessor implements Processor {	
	
	Store<Localization> localizations;
	Store<Localization> filteredLocalizations;

	int ROIx;
	int ROIxLength;
	int ROIy;
	int ROIyLength;
	
	public ROISelectProcessor(int x, int xLength, int y, int yLength) {
		ROIx=x; ROIxLength=xLength; ROIy=y; ROIyLength=yLength;
	}
	
	@Override
	public void run() {
		
		if (localizations==null) {new NullStoreWarning(this.getClass().getName()); return;}
		if (filteredLocalizations==null) {new NullStoreWarning(this.getClass().getName()); return;}
		
		Localization loc;
		while ((loc=localizations.get())!=null){
			double x = loc.getX();
			double y = loc.getY();
			if ( (x > ROIx) && (x < ROIx + ROIxLength) && (y > ROIy) && (y < ROIy + ROIyLength) ){
				filteredLocalizations.put(loc);
			}
		}
	}

	@Override
	public void setInput(Store<Localization> s) {
		localizations = s;
	}

	@Override
	public void setOutput(Store<Localization> s) {
		filteredLocalizations = s;
	}

}
