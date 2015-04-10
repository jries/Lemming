package org.lemming.inputs;

import ij.gui.GenericDialog;

import java.awt.GridBagConstraints;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import javax.swing.JButton;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.integer.UnsignedShortType;

import org.lemming.data.ImgLib2Frame;
import org.lemming.utils.LFile;
import org.lemming.utils.LemMING;

/** 
 * This DAXLoader class is used for reading frames from a dax file and feeding
 * each frame into a Frame store <p>
 * A 'dax' file contains data that is saved in a binary, 16-bit, unsigned integer
 * file format. Each 'dax' file has an 'inf' file associated with it which provides 
 * details (like the number of frames, image width, image height, byte order) that
 * are necessary to be able to read bytes from the 'dax' file. The 'inf' file must
 * have the same name as the 'dax' file (e.g. path/filename.dax &harr; path/filename.inf). 
 * 
 * @author Joe Borbely
 * 
 **/
public class DAXLoader extends SingleOutput<ImgLib2Frame<UnsignedShortType>> {

	protected int width;
	protected int height;
	protected long nFrames;
	protected String daxFilename;

	private long curFrame;
	private boolean isBigEndian;
	private boolean savedInf;
	private String infFilename;

	/**
	 * @param filename - file
	 */
	public DAXLoader(String filename) {

		// create the filename references for the dax and inf files
		LFile daxFile = new LFile(filename);
		daxFilename = filename;
		infFilename = filename.substring(0, filename.length()-3)+"inf";
    	
    	// check that the filename is a dax file
    	String ext = daxFile.getExtension();
    	if (!ext.equals("dax"))
    		LemMING.error("DAXLoader cannot open files with a ."+ext+" extension");

    	// make sure that the DAX file exists
    	if (!daxFile.exists())
    		LemMING.error("A '" + daxFile.getName() + "' file does not exist");
    	
    	// read the inf file
    	getInfo(filename);
    	
    }
	
	/** Reads the 'inf' file in order to determine the number of frames, the image 
	 *  width, the image height and the byte endianness (little or big).<p>
	 *  Checks that there is a 'inf' file associated with this 'dax' file and if the 
	 *  'inf' file does not exist then a dialog window will open to ask the user for
	 *  the parameters required to read the 'dax' file. 
	 * @param daxFilename - daxFilename must be the full file path 
	 */
	public void getInfo(String daxFilename) {
		try {						
			BufferedReader br = new BufferedReader(new FileReader(infFilename));
			String line;
			isBigEndian = false;
			try {
				while ((line = br.readLine()) != null){
					String[] lineSplit = line.split("=");
					if(lineSplit[0].trim().equals("DAX_data_type") && lineSplit[1].matches("(?i).*big.*")){
						isBigEndian = true;							
					}
					if(lineSplit[0].trim().equals("number of frames")){						
						nFrames = Integer.valueOf(lineSplit[1].trim()); 
					}					
					if(lineSplit[0].trim().equals("frame dimensions")){
						String[] temp = lineSplit[1].split("x");
						width = Integer.valueOf(temp[0].trim()); 
						height = Integer.valueOf(temp[1].trim());
					}
				}
				br.close();
			} catch (IOException e) {
				LemMING.error(e.getMessage());
			}
		} catch (FileNotFoundException e) {
			// Use the generic ImageJ dialog to ask the user what the .inf file parameters are
			final GenericDialog gd = new GenericDialog("*.inf parameter request");
			gd.addMessage("Can't find the corresponding inf file for\n"+daxFilename);
			gd.addMessage("Please specify the following parameters");
			gd.addNumericField("Frame width", width, 0);
			gd.addNumericField("Frame height", height, 0);
			gd.addNumericField("Number of frames", 1, 0);
			gd.addCheckbox("Big-endian", true);			
			JButton saveButton = new JButton("Save"); // create a button to save these settings for this dax file 
			saveButton.setToolTipText("Save these inf settings to a file for future use");
			saveButton.addActionListener(new ActionListener() {
	            public void actionPerformed(ActionEvent event) {
					width = (int) gd.getNextNumber();
					height = (int) gd.getNextNumber();
					nFrames = (int) gd.getNextNumber();
	            	isBigEndian = gd.getNextBoolean();
	            	savedInf = createInfFile();
	            	gd.dispose();
	            }
	        });
			GridBagConstraints c = new GridBagConstraints();
			c.gridx = 0; c.gridy = 5;
			c.anchor = GridBagConstraints.CENTER;
			c.insets = new Insets(15,60,0,0);
			c.gridwidth = 2;
			gd.add(saveButton, c);
			gd.centerDialog(true);
			gd.showDialog();
			if (gd.wasCanceled())
				LemMING.error("DAXLoader cannot open a dax file without a .inf file, goodbye");
			if(!savedInf){
				width = (int) gd.getNextNumber();
				height = (int) gd.getNextNumber();
				nFrames = (int) gd.getNextNumber();
	        	isBigEndian = gd.getNextBoolean();
			}
		}
	}

	/** Create a new 'inf' file for the corresponding <code>daxFilename</code> file.
	 *  @return whether the creation of the .inf file was successful
	 */
	public boolean createInfFile(){
    	try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(infFilename));
			bw.write("DAX_file_path = "+daxFilename+"\n");
			if (isBigEndian) {
				bw.write("DAX_data_type = I16_big_endian\n");					
			} else {
				bw.write("DAX_data_type = I16_little_endian\n");
			}
			bw.write("frame_X_size = "+Integer.toString(width)+"\n");
			bw.write("frame_Y_size = "+Integer.toString(height)+"\n");
			bw.write("number_of_frames = "+Long.toString(nFrames)+"\n");
			bw.write("frame dimensions = "+Integer.toString(width)+ " x "+Integer.toString(height)+"\n");
			bw.write("number of frames = "+Long.toString(nFrames)+"\n");
			bw.flush();
			bw.close();
			return true;
		} catch (IOException e) {			
			return false;			
		}
	}

	
	/** Read the data from the dax file for the specified frame number.
	 *  Automatically handles reading data from virtual stacks.
	 * 
	 * @param frame the frame number. Allowed values are 1 &le; frame &le; nFrames  
	 * @return the data for this frame or <code>null</code> if there was an error
	 */
	public short[] readFrame(long frame){
		short[] daxFrame = new short[width*height];
		byte[] bytes = new byte[width*height*2];
		RandomAccessFile raf;
		try {
			raf = new RandomAccessFile(daxFilename, "r");
			raf.seek(frame * width * height * 2);
			raf.readFully(bytes);
			raf.close();
			ByteBuffer bb = ByteBuffer.wrap(bytes); // the default endianness of ByteBuffer is big
			if(!isBigEndian) bb.order(ByteOrder.LITTLE_ENDIAN);
			for (int i=0; i < daxFrame.length; i++)
				daxFrame[i] = bb.getShort();
		} catch (FileNotFoundException e1) {
			// we have already done this check, so we will never end up here
		} catch (IOException e) {
			LemMING.error("Can\'t read dax file: "+daxFilename);
		}
		return daxFrame;
	}

    @Override
	public void beforeRun() {
       	curFrame = 0;
 	}
	
	@Override
	public boolean hasMoreOutputs() {
		return curFrame < nFrames;
	}

	@Override
	public ImgLib2Frame<UnsignedShortType> newOutput() {
		short[] pixelArray = readFrame(curFrame);
		Img<UnsignedShortType> imglib2Array = ArrayImgs.unsignedShorts(pixelArray, new long[]{width,height});
		ImgLib2Frame<UnsignedShortType> out = new ImgLib2Frame<UnsignedShortType>(curFrame, width, height, imglib2Array);
		curFrame++;
		return out;
	}

}
