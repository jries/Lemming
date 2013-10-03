package org.lemming.input;

import ij.ImagePlus;
import ij.ImageStack;
import ij.io.Opener;

import java.util.ArrayList;
import java.util.List;

import org.lemming.data.Frame;
import org.lemming.data.FrameProducer;
import org.lemming.data.Store;
import org.lemming.outputs.SO;
import org.lemming.utils.LArrays;
import org.lemming.utils.LFile;

public class TIFFLoader extends SO {

	Store<Frame> s; 
	LFile file; 
	ImagePlus imp;
	
    public TIFFLoader(String filename) {
    	
    	// create the filename reference
    	file = new LFile(filename);

    	// check that the filename is a TIFF file
    	List<String> extns = new ArrayList<String>(LArrays.asList("tiff", "tif"));
    	String ext = file.getExtension();
    	if (!extns.contains(ext))
    		throw new RuntimeException("TIFFLoader cannot open files with a ."+ext+" extension");

    	// make sure that the TIFF file exists
    	if (!file.exists())
    		throw new RuntimeException("A '" + filename + "' file does not exist");    
    }

    @Override
	public void run() {
    	imp = new Opener().openImage(file.getAbsolutePath());
    	ImageStack stack =imp.getStack();
        for (int i=1, n=stack.getSize(); i<=n; i++){
        	s.put(new FrameProducer(stack.getPixels(i)));        	
        }        
 	}

	@Override
	public void setOutput(Store<Frame> store) {
		s = store;
	}

	@Override
	public boolean hasMoreFrames() {
		return !s.isEmpty();
	}

	@Override
	public Frame newFrame() {
		return null;
	}
	
	/** Display the TIFF file */
	public void show() {
		imp.show();		
	}

}
