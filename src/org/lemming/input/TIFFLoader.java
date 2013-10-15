package org.lemming.input;

import java.util.ArrayList;
import java.util.List;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.io.ImgIOException;
import net.imglib2.io.ImgOpener;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.lemming.data.Frame;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.Store;
import org.lemming.outputs.SO;
import org.lemming.utils.LArrays;
import org.lemming.utils.LFile;

public class TIFFLoader extends SO<Frame> {

	Store<Frame> s; 
	LFile file; 
	
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

    Img< FloatType > imageFloat;
    long curSlice;
    
    @Override
	public void run() {
        try {
			imageFloat = new ImgOpener().openImg( file.getAbsolutePath(),
			        new ArrayImgFactory< FloatType >(), new FloatType() );

			curSlice = 0;

			super.run();
			
		} catch (ImgIOException e) {
			e.printStackTrace();
		}
 	}

	@Override
	public boolean hasMoreOutputs() {
		return curSlice < imageFloat.dimension(2);
	}

	@Override
	public Frame newOutput() {
		Frame out = new ImgLib2Frame(curSlice, (int)imageFloat.dimension(0), (int)imageFloat.dimension(1), Views.hyperSlice(imageFloat, 2, curSlice)); 
		curSlice++;
		return out;
	}
	
	/** Display the TIFF file */
	public void show() {
		ImageJFunctions.show( imageFloat );		
	}

}
